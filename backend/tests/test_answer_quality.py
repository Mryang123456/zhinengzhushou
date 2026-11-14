"""证据校验、工具失败和被拒草稿隔离。"""

from unittest.mock import patch
import asyncio
import json
import os
import unittest
from datetime import date as Date, timedelta

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import ToolException

from backend.app.agent.service import build_agent
from backend.app.agent.tools import calculator
from backend.app.agent.turns import checkpoint_config, stream_events
from backend.app.quality.validator import validate_answer


class AnswerQualityTests(unittest.TestCase):
    def test_current_weather_does_not_match_day_before_yesterday_marker(self):
        report = "上海当前天气：晴朗；气温 12°C，体感 10°C。"
        messages = [ToolMessage(content=report, name="get_weather", tool_call_id="weather")]
        self.assertTrue(validate_answer("上海当前天气怎么样", report, messages)["ok"])
        self.assertFalse(validate_answer("上海前天天气怎么样", report, messages)["ok"])

    def test_compound_weather_question_corrects_model_date_and_answers_clothing(self):
        question = "哈喽你好 请帮我查询明天杭州的天气，并且告诉我适合去旅游吗，穿什么衣服合适"
        tomorrow = (Date.today() + timedelta(days=1)).isoformat()
        report = f"杭州天气预报（{tomorrow}）：小雨；最高气温 21°C，最低气温 14°C，最低体感 13°C。"

        class Model:
            def bind_tools(self, tools):
                return self

            async def astream(self, messages):
                yield AIMessageChunk(content="", tool_calls=[
                    {"name": "get_weather", "id": "weather", "args": {"location": "杭州", "date": "current"}},
                ])

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=question)
            return [frame async for frame in stream_events(graph, question, "weather-and-clothing")]

        with patch("backend.app.agent.service.create_chat_model", return_value=Model()):
            with patch("backend.app.agent.tools.query_weather", return_value=report) as fetch:
                events = asyncio.run(collect())
                self.assertEqual(fetch.call_args.args[:3], ("杭州", "明天", 1))
            self.assertIn("不太适合长时间在户外", events[-1]["content"])
            self.assertIn("穿衣建议：建议穿长袖和长裤", events[-1]["content"])
            self.assertIn(report, events[-1]["content"])
            with patch("backend.app.agent.tools.query_weather", side_effect=TimeoutError):
                events = asyncio.run(collect())
            self.assertIn("天气查询未成功", events[-1]["content"])
            self.assertNotIn("穿衣建议", events[-1]["content"])

    def test_model_receives_verified_weather_and_user_context_for_advice(self):
        question = "明天杭州天气适合带孩子去旅游吗，穿什么衣服合适"
        tomorrow = (Date.today() + timedelta(days=1)).isoformat()
        report = f"杭州天气预报（{tomorrow}）：小雨；最高气温 21°C，最低气温 14°C。"
        advice_inputs = []

        class Model:
            def bind_tools(self, tools):
                return self

            async def astream(self, messages):
                if messages[-1].content == question:
                    yield AIMessageChunk(content="", tool_calls=[
                        {"name": "get_weather", "id": "weather", "args": {"location": "杭州", "date": "明天"}},
                    ])
                    return
                advice_inputs.append(json.loads(messages[-1].content))
                yield AIMessageChunk(content=json.dumps({
                    "activity": "可以出游，带孩子优先安排有室内休息区的活动。",
                    "clothing": "给孩子带一件可随时穿脱的外套，另备干袜子。",
                }, ensure_ascii=False))

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=question)
            return [frame async for frame in stream_events(graph, question, "weather-model-advice")]

        with (
            patch("backend.app.agent.service.create_chat_model", return_value=Model()),
            patch("backend.app.agent.tools.query_weather", return_value=report),
        ):
            events = asyncio.run(collect())

        self.assertEqual(advice_inputs, [{
            "用户原问题": question, "已核验预报": report, "需要出游判断": True, "需要穿衣建议": True,
        }])
        self.assertIn("已采用模型建议", next(frame["reason"] for frame in events if frame["type"] == "quality"))
        self.assertIn("另备干袜子", events[-1]["content"])
        self.assertTrue(events[-1]["content"].endswith(report))

    def test_weather_suitability_answers_the_decision_using_the_matching_forecast(self):
        from backend.app.config import SKILLS_DIR
        from backend.app.intent.policy import classify
        from backend.app.skills.registry import load_skills

        skills = load_skills(SKILLS_DIR)
        question = "明天杭州天气适合旅游吗"
        self.assertTrue(classify(question, skills)["weather_assessment"])
        self.assertTrue(classify("明天厦门下雨还能露营吗", skills)["weather_assessment"])
        self.assertFalse(classify("明天杭州天气怎么样", skills).get("weather_assessment"))
        tomorrow = (Date.today() + timedelta(days=1)).isoformat()

        class FakeModel:
            def bind_tools(self, tools):
                if [tool.name for tool in tools] != ["get_weather"]:
                    raise AssertionError("活动判断只开放天气工具")
                return self

            async def astream(self, messages):
                yield AIMessageChunk(content="", tool_calls=[
                    {"name": "get_weather", "id": "weather", "args": {"location": "杭州", "date": "明天"}},
                ])

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=question)
            return [frame async for frame in stream_events(graph, question, "weather-assessment")]

        for city, expected in (("杭州", "不太适合长时间在户外"), ("上海", "暂时无法判断")):
            report = f"{city}天气预报（{tomorrow}）：中等阵雨；最高气温 20°C，最大降水概率 100%。"
            with (
                self.subTest(city=city),
                patch("backend.app.agent.service.create_chat_model", return_value=FakeModel()),
                patch("backend.app.agent.tools.query_weather", return_value=report),
            ):
                events = asyncio.run(collect())
            self.assertIn(expected, events[-1]["content"])
            self.assertEqual(report in events[-1]["content"], city == "杭州")

    def test_quality_gate_blocks_unverified_freshness_and_high_risk(self):
        self.assertFalse(validate_answer("今天北京天气", "晴朗", [])["ok"])
        self.assertFalse(validate_answer("应该买哪只股票", "买入 A", [])["ok"])
        self.assertTrue(validate_answer("解释 Python 列表", "列表是有序集合。", [])["ok"])
        self.assertFalse(validate_answer("解释配置", "请使用 api_key=very-secret-value-123456", [])["ok"])

    def test_quality_gate_requires_inline_evidence_for_factual_web_claims(self):
        source = json.dumps([
            {"href": "https://example.com/report", "body": "数据为 42。"},
            {"href": "https://example.com/other", "body": "样本数为 900。"},
        ], ensure_ascii=False)
        messages = [ToolMessage(source, name="web_search", tool_call_id="web")]
        self.assertTrue(validate_answer("调研数据", "数据为 42 https://example.com/report", messages)["ok"])
        for answer in (
            "数据为 900 https://example.com/report",
            "样本数为 900 https://example.com/report",
            "数据为 42 https://example.com/report\n结论可靠。",
            "数据为 42 https://example.com/report https://example.com/other",
        ):
            with self.subTest(answer=answer):
                self.assertFalse(validate_answer("调研数据", answer, messages)["ok"])

    def test_single_news_source_allows_cited_summary_with_verified_numbers(self):
        source = "https://example.com/news"
        message = ToolMessage(json.dumps({
            "url": source, "title": "发布会消息", "body": "黄仁勋介绍了 2 款新品。", "date": "2026-11-13",
        }, ensure_ascii=False), name="search_search_news", tool_call_id="news")
        summary = f"据这篇报道，黄仁勋展示了 2 款新产品（2026-11-13）。[报道链接]({source})"
        self.assertTrue(validate_answer("黄仁勋最近的新闻", summary, [message])["ok"])
        for question, answer, tool_message in (
            ("最近新闻", summary.replace("2 款", "3 款"), message),
            ("最近新闻", summary.replace("11-13", "11-14"), message),
            ("最近新闻", summary.replace(source, "https://example.com/unknown"), message),
            ("最近新闻", summary, message.model_copy(update={"status": "error"})),
            ("最近政策", summary, message),
            ("应该买哪只股票", summary, message),
        ):
            with self.subTest(question=question, answer=answer):
                self.assertFalse(validate_answer(question, answer, [tool_message])["ok"])

    def test_title_only_result_can_be_listed_but_does_not_support_a_summary(self):
        link = "https://example.com/headline"
        message = ToolMessage(json.dumps({"url": link, "title": "发布会预告"}, ensure_ascii=False),
                              name="search_search_news", tool_call_id="headline")
        self.assertTrue(validate_answer("最近新闻", f"发布会预告 [链接]({link})", [message])["ok"])
        self.assertFalse(validate_answer("最近新闻", f"发布会已经举行 [链接]({link})", [message])["ok"])

    def test_news_summary_accepts_separate_source_lines_and_keeps_item_boundaries(self):
        first = "https://example.com/a"
        second = "https://example.org/b"
        message = ToolMessage(json.dumps([
            {"url": first, "body": "会上展示了 2 款新品。"},
            {"url": second, "body": "会上公布了 3 个合作项目。"},
        ], ensure_ascii=False), name="search_search_news", tool_call_id="news")
        answer = (
            "## 近期新闻\n\n1. **发布会**\n展示了 2 款新品。\n"
            f"来源：[第一篇报道]({first})\n\n"
            f"2. 公布了 3 项合作。\n\n**来源：** [第二篇报道]({second})"
        )
        self.assertTrue(validate_answer("最近新闻", answer, [message])["ok"])
        for invalid in (
            answer.replace(first, "https://example.net/fake"),
            answer.replace(first, second),
            answer + "\n\n3. 还公布了新计划。",
        ):
            self.assertFalse(validate_answer("最近新闻", invalid, [message])["ok"])

    def test_news_graph_falls_back_to_verified_search_links(self):
        question = "帮我查询黄仁勋最近的新闻"
        source = "https://example.com/news"
        fabricated = "https://example.org/unknown"

        class FakeModel:
            def bind_tools(self, _tools):
                return self

            async def astream(self, messages):
                if isinstance(messages[-1], ToolMessage):
                    yield AIMessageChunk(content=f"黄仁勋展示了新产品。[报道]({fabricated})")
                    return
                yield AIMessageChunk(content="", tool_calls=[
                    {"name": "web_search", "id": "news", "args": {"query": "黄仁勋 新闻"}},
                ])

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=question)
            return [frame async for frame in stream_events(graph, question, "verified-news")]

        with (
            patch.dict(os.environ, {"KNOWLEDGE_DATABASE_URL": "", "TOOL_RETRIEVAL_EMBEDDING_PROVIDER": ""}),
            patch("backend.app.agent.service.create_chat_model", return_value=FakeModel()),
            patch("backend.app.agent.tools.DDGS") as search,
        ):
            search.return_value.text.return_value = [{"href": source, "body": "黄仁勋在发布会上介绍了新品。"}]
            events = asyncio.run(collect())
        self.assertEqual([event["passed"] for event in events if event["type"] == "quality"], [True])
        self.assertIn(source, events[-1]["content"])
        self.assertNotIn(fabricated, "".join(event["content"] for event in events if event["type"] == "token"))

    def test_event_answer_repairs_draft_into_prose_with_server_rendered_citations(self):
        question = "星河科技为什么裁员"
        link = "https://example.com/announcement"
        drafts = []

        class FakeModel:
            def bind_tools(self, _tools):
                return self

            async def astream(self, messages):
                drafts.append(messages)
                if len(drafts) == 1:
                    yield AIMessageChunk(content="这是未经引用的草稿。")
                else:
                    yield AIMessageChunk(content="据公司公告，星河科技已经启动裁员。公告说明了人员调整，但没有交代具体原因。[S1]")

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=question)
            return [event async for event in stream_events(graph, question, "prose-news-repair")]

        with (
            patch("backend.app.agent.service.create_chat_model", return_value=FakeModel()),
            patch("backend.app.agent.tools.DDGS") as search,
        ):
            search.return_value.text.return_value = [{"href": link, "body": "公司公告称已开始裁员，未披露原因。"}]
            events = asyncio.run(collect())
            search.return_value.text.assert_called_once()
        self.assertEqual(len(drafts), 2)
        self.assertIn("上一版校验原因", drafts[1][-1].content)
        answer = events[-1]["content"]
        self.assertIn("没有交代具体原因", answer)
        self.assertIn(f"[来源 1]({link})", answer)
        self.assertNotIn("未经引用的草稿", "".join(event["content"] for event in events if event["type"] == "token"))

    def test_quality_gate_rejects_failed_empty_and_unusable_evidence(self):
        weather_error = ToolMessage("气温 31°C", name="get_weather", tool_call_id="weather", status="error")
        empty_weather = ToolMessage("未找到指定地点", name="get_weather", tool_call_id="empty-weather")
        weather_ok = ToolMessage("北京当前天气：晴朗；气温 18°C", name="get_weather", tool_call_id="weather-ok")
        web_error = ToolMessage("https://example.com/failed", name="web_search", tool_call_id="web", status="error")
        web_ok = ToolMessage('{"href":"https://example.com/report","body":"北京新开一家图书馆。"}', name="search_search_text", tool_call_id="web-ok")
        kb_empty = ToolMessage("", name="search_knowledge_base", tool_call_id="kb")

        for messages in ([weather_error], [empty_weather], [web_error]):
            self.assertFalse(validate_answer("今天北京天气", "北京今天晴朗", messages)["ok"])
        self.assertTrue(validate_answer("今天北京天气", weather_ok.content, [web_error, weather_ok])["ok"])
        self.assertFalse(validate_answer("今天北京天气", "北京气温 31°C https://example.com/report", [weather_error, web_ok])["ok"])
        self.assertFalse(validate_answer("今天北京天气", "北京当前天气：晴朗；气温 31°C", [weather_ok])["ok"])
        self.assertFalse(validate_answer("今天北京天气", "北京气温 18°C https://example.com/failed", [web_error, weather_ok])["ok"])
        self.assertFalse(validate_answer("调研数据", "结果为 42 https://example.com/failed", [web_error, web_ok])["ok"])
        self.assertFalse(validate_answer("内部手册怎么说", "按流程操作。", [kb_empty])["ok"])
        self.assertFalse(validate_answer("内部手册怎么说", "依据 knowledge://7#2，按流程操作。", [])["ok"])

    def test_weather_report_must_match_requested_city_and_current_time(self):
        beijing = ToolMessage("北京当前天气：晴朗；气温 18°C。", name="get_weather", tool_call_id="beijing")
        tokyo = ToolMessage("东京当前天气：晴朗；气温 18°C。", name="weather_current_weather", tool_call_id="tokyo")
        self.assertTrue(validate_answer("今天北京天气", beijing.content, [beijing, tokyo])["ok"])
        self.assertTrue(validate_answer("东京现在天气怎么样", tokyo.content, [tokyo])["ok"])
        for question, report, messages in (
            ("今天北京天气", tokyo.content, [tokyo]),
            ("今天北京天气", tokyo.content, [beijing, tokyo]),
            ("明天北京天气", beijing.content, [beijing]),
            ("北京今天下午的气温", beijing.content, [beijing]),
            ("北京 12 月 1 日天气", beijing.content, [beijing]),
        ):
            with self.subTest(question=question):
                self.assertFalse(validate_answer(question, report, messages)["ok"])

    def test_weather_forecast_must_match_location_and_requested_date(self):
        tomorrow = (Date.today() + timedelta(days=1)).isoformat()
        day_after = (Date.today() + timedelta(days=2)).isoformat()
        hangzhou = ToolMessage(
            f"杭州天气预报（{tomorrow}）：小雨；最高气温 21°C，最低气温 12°C。",
            name="get_weather", tool_call_id="hangzhou",
        )
        wrong_date = ToolMessage(
            f"杭州天气预报（{day_after}）：晴朗；最高气温 23°C，最低气温 13°C。",
            name="get_weather", tool_call_id="wrong-date",
        )
        tokyo = ToolMessage(
            f"东京天气预报（{tomorrow}）：晴朗；最高气温 20°C，最低气温 10°C。",
            name="weather_current_weather", tool_call_id="tokyo",
        )
        self.assertTrue(validate_answer("明天杭州天气", hangzhou.content, [hangzhou])["ok"])
        self.assertTrue(validate_answer(f"杭州 {tomorrow} 天气", hangzhou.content, [hangzhou])["ok"])
        self.assertFalse(validate_answer("明天杭州天气", wrong_date.content, [wrong_date])["ok"])
        self.assertFalse(validate_answer("明天杭州天气", tokyo.content, [tokyo])["ok"])
        self.assertFalse(validate_answer("明天下午杭州天气", hangzhou.content, [hangzhou])["ok"])

    def test_weather_tool_failure_reaches_quality_gate_in_real_graph(self):
        test = self

        class FakeModel:
            def bind_tools(self, tools):
                test.assertIn("get_weather", [item.name for item in tools])
                return self

            async def astream(self, messages):
                tool_results = [msg for msg in messages if isinstance(msg, ToolMessage)]
                if not tool_results:
                    yield AIMessageChunk(content="", tool_calls=[
                        {"id": "weather", "name": "get_weather", "args": {"location": "北京"}},
                    ])
                    return
                test.assertEqual(tool_results[-1].status, "error")
                yield AIMessageChunk(content="北京今天晴朗，气温 31°C。")

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question="今天北京天气")
            return [frame async for frame in stream_events(graph, "今天北京天气", "failed-weather")]

        with (
            patch.dict(os.environ, {"KNOWLEDGE_DATABASE_URL": "", "TOOL_RETRIEVAL_EMBEDDING_PROVIDER": ""}),
            patch("backend.app.agent.service.create_chat_model", return_value=FakeModel()),
            patch("backend.app.agent.tools._get_json", side_effect=TimeoutError),
        ):
            events = asyncio.run(collect())
        self.assertEqual(events[-1]["type"], "done")
        self.assertIn("暂时无法确认", events[-1]["content"])
        self.assertNotIn("31°C", events[-1]["content"])

    def test_rejected_weather_draft_uses_only_the_matching_tool_forecast(self):
        question = "帮我查询明天杭州的天气"
        tomorrow = (Date.today() + timedelta(days=1)).isoformat()
        day_after = (Date.today() + timedelta(days=2)).isoformat()
        matching = f"杭州天气预报（{tomorrow}）：小雨；最高气温 22°C，最低气温 12°C。"
        wrong_date = f"杭州天气预报（{day_after}）：晴朗；最高气温 25°C。"
        wrong_city = f"北京天气预报（{tomorrow}）：晴朗；最高气温 23°C。"
        draft = "明天杭州会下雨，最高 99°C。"

        class FakeModel:
            def bind_tools(self, _tools):
                return self

            async def astream(self, messages):
                if isinstance(messages[-1], ToolMessage):
                    yield AIMessageChunk(content=draft)
                    return
                yield AIMessageChunk(content="", tool_calls=[
                    {"id": "weather", "name": "get_weather", "args": {"location": "杭州", "date": "明天"}},
                ])

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=question)
            return [frame async for frame in stream_events(graph, question, "verified-weather")]

        for report, expected in (
            ("\n".join([wrong_date, matching, wrong_city]), matching),
            (wrong_date, "天气工具没有取得与地点和日期匹配的有效预报，暂时无法确认。"),
            (wrong_city, "天气工具没有取得与地点和日期匹配的有效预报，暂时无法确认。"),
        ):
            with (
                self.subTest(report=report),
                patch.dict(os.environ, {"KNOWLEDGE_DATABASE_URL": "", "TOOL_RETRIEVAL_EMBEDDING_PROVIDER": ""}),
                patch("backend.app.agent.service.create_chat_model", return_value=FakeModel()),
                patch("backend.app.agent.tools.query_weather", return_value=report),
            ):
                events = asyncio.run(collect())
                self.assertEqual(events[-1], {"type": "done", "content": expected})
                self.assertEqual("".join(event["content"] for event in events if event["type"] == "token"), expected)
                self.assertEqual([event["passed"] for event in events if event["type"] == "quality"], [expected == matching])

    def test_rejected_draft_is_absent_from_checkpoint_history_and_next_prompt(self):
        draft = "当前数值为 900"
        test = self

        class FakeModel:
            async def astream(self, messages):
                if messages[-1].content == "当前数值是多少":
                    yield AIMessageChunk(content=draft)
                    return
                test.assertFalse(any(draft in str(message.content) for message in messages))
                yield AIMessageChunk(content="这是后续回复")

        async def collect():
            graph = await build_agent(memory_db_path=None)
            first = [frame async for frame in stream_events(graph, "当前数值是多少", "rejected-draft")]
            self.assertIn("暂时无法确认", first[-1]["content"])
            config = checkpoint_config("rejected-draft")
            async for checkpoint in graph.aget_state_history(config):
                self.assertFalse(any(draft in str(message.content) for message in checkpoint.values.get("messages", [])))
            snapshot = await graph.aget_state(config)
            self.assertIsNone(snapshot.values["draft_answer"])
            second = [frame async for frame in stream_events(graph, "继续说明", "rejected-draft")]
            self.assertEqual(second[-1]["content"], "这是后续回复")

        with patch("backend.app.agent.service.create_chat_model", return_value=FakeModel()):
            asyncio.run(collect())

    def test_legacy_checkpoint_discards_rejected_draft_and_resets_summary(self):
        draft = "当前数值为 900"
        test = self

        class FakeModel:
            async def astream(self, messages):
                test.assertFalse(any(draft in str(message.content) for message in messages))
                test.assertTrue(any(message.content == "暂时无法确认" for message in messages))
                yield AIMessageChunk(content="这是后续回复")

        async def collect():
            graph = await build_agent(memory_db_path=None)
            config = checkpoint_config("legacy-rejected-draft")
            await graph.aupdate_state(config, {
                "messages": [
                    HumanMessage("当前数值是多少"),
                    AIMessage(draft),
                    AIMessage("暂时无法确认"),
                ],
                "conversation_summary": draft,
                "summary_message_count": 3,
            }, as_node="output")
            frames = [frame async for frame in stream_events(graph, "继续说明", "legacy-rejected-draft")]
            self.assertEqual(frames[-1]["content"], "这是后续回复")
            snapshot = await graph.aget_state(config)
            self.assertEqual(snapshot.values["conversation_summary"], "")
            self.assertNotIn(draft, [message.content for message in snapshot.values["messages"]])

        with patch("backend.app.agent.service.create_chat_model", return_value=FakeModel()):
            asyncio.run(collect())

    def test_successful_calculation_replaces_wrong_model_result_in_real_graph(self):
        class FakeModel:
            def bind_tools(self, tools):
                return self

            async def astream(self, messages):
                if isinstance(messages[-1], ToolMessage):
                    yield AIMessageChunk(content="结果是 41。")
                else:
                    yield AIMessageChunk(content="", tool_calls=[
                        {"name": "calculator", "id": "calc", "args": {"expression": "(17 + 3) * 2"}},
                    ])

        async def collect():
            graph = await build_agent(memory_db_path=None)
            return [frame async for frame in stream_events(graph, "(17 + 3) * 2", "checked-calculation")]

        with patch("backend.app.agent.service.create_chat_model", return_value=FakeModel()):
            frames = asyncio.run(collect())
        self.assertEqual(frames[-1], {"type": "done", "content": "计算结果：40"})
        self.assertFalse(any("41" in frame.get("content", "") for frame in frames))
        result = ToolMessage("40", name="calculator", tool_call_id="calc")
        self.assertTrue(validate_answer("计算", "结果是 40。", [result])["ok"])
        self.assertFalse(validate_answer("计算", "结果不是 40。", [result])["ok"])

    def test_failed_calculator_result_blocks_fabricated_answer_in_real_graph(self):
        test = self

        class FakeModel:
            def bind_tools(self, tools):
                test.assertIn("calculator", [tool.name for tool in tools])
                return self

            async def astream(self, messages):
                results = [message for message in messages if isinstance(message, ToolMessage)]
                if not results:
                    yield AIMessageChunk(content="", tool_calls=[
                        {"name": "calculator", "id": "divide", "args": {"expression": "1/0"}},
                        {"name": "calculator", "id": "add", "args": {"expression": "1+0"}},
                    ])
                    return
                test.assertEqual([message.status for message in results], ["error", "success"])
                yield AIMessageChunk(content="两项的计算结果都是 1。")

        async def collect():
            graph = await build_agent(memory_db_path=None)
            return [frame async for frame in stream_events(graph, "1/0", "failed-calculation")]

        with patch("backend.app.agent.service.create_chat_model", return_value=FakeModel()):
            frames = asyncio.run(collect())
        self.assertEqual(frames[-1]["type"], "done")
        self.assertIn("未能完成计算", frames[-1]["content"])
        self.assertFalse(any("都是 1" in frame.get("content", "") for frame in frames))
        for expression in ("1/0", "2+", "1e309", "(-1)**0.5"):
            with self.subTest(expression=expression), self.assertRaises(ToolException):
                calculator.invoke({"expression": expression})
        self.assertEqual(calculator.invoke({"expression": "(17 + 3) * 2"}), "40")
