"""内置工具、意图、技能和工具配额。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import asyncio
import os
import unittest

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from backend.app.agent.service import build_agent
from backend.app.agent.tool_retrieval import retrieve_tool_candidates
from backend.app.agent.tools import evaluate_expression, weather_code_description
from backend.app.agent.turns import stream_events
from backend.app.config import SKILLS_DIR
from backend.app.intent.policy import classify
from backend.app.models.factory import ModelConfigurationError, get_model_settings
from backend.app.skills.registry import (
    load_skills,
    match_skills,
    select_tool_names,
    select_top_tool_candidates,
)


class ToolRoutingTests(unittest.TestCase):
    def test_web_search_caps_each_query_at_fifty_results(self):
        import json
        from backend.app.agent.tools import web_search

        results = [
            {"href": f"https://example.org/{index}", "title": f"结果 {index}", "body": "相关信息"}
            for index in range(55)
        ]
        with patch("backend.app.agent.tools.DDGS") as search:
            search.return_value.text.return_value = results
            records = json.loads(web_search.invoke({"query": "查询测试"}))

        search.return_value.text.assert_called_once_with("查询测试", max_results=50)
        self.assertEqual(len(records), 50)

    def test_web_result_budget_deduplicates_parallel_searches_and_reserves_retry(self):
        import json
        from backend.app.quality.validator import web_result_count
        from backend.app.quality.web_result_budget import select_web_results

        def result(name, call_id, indices):
            return ToolMessage(json.dumps([{"href": f"https://example.org/{i}", "body": f"正文 {i}"} for i in indices]),
                               name=name, tool_call_id=call_id)

        batch = [result("web_search", "a", range(40)), result("search_search_text", "b", range(29, 70))]
        first = select_web_results([], batch, reserve_retry=True)
        self.assertEqual(sum(map(len, first.values())), 45)
        self.assertTrue(first["a"] and first["b"])
        previous = [message.model_copy(update={"content": json.dumps(first[message.tool_call_id])}) for message in batch]
        followup = result("web_search", "c", range(100))
        second = select_web_results(previous, [followup], reserve_retry=False)
        self.assertEqual(len(second["c"]), 5)
        saved = followup.model_copy(update={"content": json.dumps(second["c"])})
        self.assertEqual(web_result_count([*previous, saved]), 50)

    def test_web_answer_sources_keeps_all_unique_search_sites(self):
        import json
        from backend.app.quality.validator import web_answer_sources

        results = [
            {"href": f"https://example.org/{index}", "title": f"来源 {index}", "body": f"证据 {index}"}
            for index in range(15)
        ]
        message = ToolMessage(json.dumps(results, ensure_ascii=False), name="web_search", tool_call_id="all-sites")

        sources = web_answer_sources("查询来源", [message])

        self.assertEqual(len(sources), 15)

    def test_search_synthesizes_late_passages_into_a_cited_markdown_answer(self):
        import json
        from backend.app.intent.semantic import SemanticIntent

        question = "海岚 A2 有线充电和无线充电分别是多少瓦"
        wired_url = "https://example.org/wired"
        wireless_url = "https://example.com/wireless"
        wired = "海岚 A2 的有线充电功率为 45 瓦。"
        wireless = "海岚 A2 支持 15 瓦无线充电。"
        boilerplate = "".join(f"公司团队在展厅合影，内部活动编号 {index}。" for index in range(75))
        test = self
        checked = {"review": False, "draft": False}

        class Model:
            async def ainvoke(self, messages):
                results = json.loads(messages[-1].content)["结果"]
                test.assertEqual([item["编号"] for item in results], ["S1", "S2"])
                test.assertIn(wired, results[0]["摘录"])
                test.assertIn(wireless, results[1]["摘录"])
                checked["review"] = True
                return AIMessage(content='{"action":"answer"}')

            async def astream(self, messages):
                excerpts = json.loads(messages[-1].content)
                test.assertEqual(excerpts[0]["网站"], "example.org")
                test.assertIn(wired, excerpts[0]["摘录"])
                test.assertIn(wireless, excerpts[1]["摘录"])
                test.assertNotIn("内部活动编号 12", messages[-1].content)
                checked["draft"] = True
                yield AIMessageChunk(content=(
                    "**有线 45 瓦，无线 15 瓦。**[S1][S2]\n\n"
                    "### 充电规格\n\n| 方式 | 功率 |\n| --- | --- |\n"
                    "| 有线 | 45 瓦 [S1] |\n| 无线 | 15 瓦 [S2] |"
                ))

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=question)
            return [event async for event in stream_events(graph, question, "search-late-passages")]

        with (
            patch.dict(os.environ, {"AGENT_DEEPSEEK_FETCH_ENABLED": ""}),
            patch("backend.app.agent.service.create_chat_model", return_value=Model()),
            patch("backend.app.agent.service.resolve_intent", new=AsyncMock(return_value=SemanticIntent(
                route="search", question=question, query=question,
            ))),
            patch("backend.app.agent.tools.DDGS") as search,
        ):
            search.return_value.text.return_value = [
                {"href": wired_url, "body": boilerplate + wired},
                {"href": wireless_url, "body": boilerplate + wireless},
            ]
            events = asyncio.run(collect())
            search.return_value.text.assert_called_once()

        self.assertTrue(all(checked.values()))
        self.assertEqual([event["queries"] for event in events if event["type"] == "web_queries"], [[question]])
        highlights = next(event["pages"] for event in events if event["type"] == "web_highlights")
        self.assertEqual([page["url"] for page in highlights], [wired_url, wireless_url])
        self.assertIn(wired, highlights[0]["excerpt"])
        self.assertIn(wireless, highlights[1]["excerpt"])
        self.assertFalse(any(event["type"] == "web_pages_read" for event in events))
        answer = events[-1]["content"]
        self.assertTrue(answer.startswith(f"**有线 45 瓦，无线 15 瓦。**[来源 1]({wired_url})[来源 2]({wireless_url})"))
        self.assertIn("### 充电规格", answer)
        self.assertIn(f"| 无线 | 15 瓦 [来源 2]({wireless_url}) |", answer)
        self.assertFalse(any(event.get("stage") in {"checking", "revising"} for event in events))

    def test_web_search_output_sanitization_does_not_truncate_large_result_sets(self):
        import json
        from backend.app.agent.service import sanitize_web_search_output

        payload = json.dumps([
            {"href": f"https://example.org/{index}", "body": "evidence " * 120}
            for index in range(20)
        ])

        sanitized = sanitize_web_search_output(payload)

        self.assertGreater(len(payload), 12_000)
        self.assertEqual(sanitized, payload)

    def test_optional_fetch_enriches_only_public_results_and_keeps_search_on_failure(self):
        import json
        from backend.app.quality.web_fetch import enrich_search_results

        valid = "https://example.org/cast"
        unreachable = "https://example.com/cast"
        content = json.dumps([
            {"href": "http://127.0.0.1/private", "body": "本地内容"},
            {"href": valid, "body": "演员表"},
            {"href": unreachable, "body": "剧情介绍"},
            {"href": "https://example.net/extra", "body": "其他"},
        ], ensure_ascii=False)
        with (
            patch.dict(os.environ, {"AGENT_DEEPSEEK_FETCH_ENABLED": "true", "DEEPSEEK_API_KEY": "fixture"}),
            patch("backend.app.quality.web_fetch.fetch_page_text", side_effect=lambda url, _: "朱丽由李念饰演。" if url == valid else "") as fetch,
        ):
            pages = []
            records = json.loads(asyncio.run(enrich_search_results(content, "朱丽 扮演者", read_pages=pages)))
        self.assertCountEqual([call.args[0] for call in fetch.call_args_list], [
            valid, unreachable, "https://example.net/extra",
        ])
        self.assertIn("朱丽由李念饰演", records[1]["body"])
        self.assertEqual(records[2]["body"], "剧情介绍")
        self.assertEqual(pages, [{"url": valid, "title": "example.org", "domain": "example.org"}])

    def test_incomplete_coverage_requeries_before_revision(self):
        import json

        cases = [
            ("《都挺好》苏明成妻子的扮演者是谁", "都挺好 苏明成 妻子", "都挺好 朱丽 扮演者 演员表",
             "苏明成的妻子是朱丽。", "朱丽由李念饰演。"),
            ("海岚 A2 支持多大功率的有线充电", "海岚 A2 产品", "海岚 A2 有线充电 规格",
             "海岚 A2 已经发布。", "海岚 A2 支持 45 瓦有线充电。"),
            ("远山博物馆周二几点闭馆", "远山博物馆", "远山博物馆 周二 闭馆 开放时间",
             "远山博物馆展出陶器。", "远山博物馆周二17:00闭馆。"),
        ]
        link = "https://example.org/source"

        for question, first, retry, clue, fact in cases:
            with self.subTest(question=question):
                class Model:
                    evidence_reviews = 0

                    async def ainvoke(self, messages):
                        data = json.loads(messages[-1].content)
                        if "当前请求" in data:
                            return AIMessage(content=json.dumps({"route": "search", "question": question, "query": first}))
                        self.evidence_reviews += 1
                        if self.evidence_reviews == 1:
                            return AIMessage(content='{"action":"partial","missing":"缺少直接支持答案的资料"}')
                        return AIMessage(content='{"action":"answer"}')

                    async def astream(self, messages):
                        text = messages[-1].content
                        if "校验反馈" in text:
                            yield AIMessageChunk(content=json.dumps({"query": retry}))
                        elif "上一版校验原因" in text:
                            yield AIMessageChunk(content="现有资料没有直接回答。")
                        elif any(isinstance(item, ToolMessage) and fact in item.content for item in messages):
                            yield AIMessageChunk(content=f"{fact}[S1]")
                        else:
                            yield AIMessageChunk(content="现有资料没有直接回答。")

                async def collect():
                    graph = await build_agent(memory_db_path=None, initial_question=question)
                    return [frame async for frame in stream_events(graph, question, "targeted-retry")]

                with (
                    patch.dict(os.environ, {"AGENT_DEEPSEEK_FETCH_ENABLED": ""}),
                    patch("backend.app.agent.service.create_chat_model", return_value=Model()),
                    patch(
                        "backend.app.agent.service.enrich_search_results",
                        new=AsyncMock(side_effect=lambda content, _query, *, read_pages: content),
                    ) as enrich,
                    patch("backend.app.agent.tools.DDGS") as search,
                ):
                    search.return_value.text.side_effect = [[{"href": link, "body": clue}], [{"href": link, "body": fact}]]
                    events = asyncio.run(collect())
                    self.assertEqual([call.args[0] for call in search.return_value.text.call_args_list], [first, retry])
                    self.assertEqual([call.args[1] for call in enrich.await_args_list], [first, retry])
                self.assertEqual(sum(frame["type"] == "evidence_retry" for frame in events), 1)
                self.assertFalse(any(
                    frame.get("stage") == "revising" for frame in events
                ))
                self.assertIn(fact, events[-1]["content"])
                self.assertIn(link, events[-1]["content"])

    def test_broad_local_food_question_searches_and_formats_cited_recommendations(self):
        question = "查询 北京有什么美食"
        link = "https://example.org/beijing-food"
        prompts = []

        class Model:
            def bind_tools(self, _tools):
                return self

            async def ainvoke(self, _messages):
                return AIMessage(content='{"action":"answer"}')

            async def astream(self, messages):
                prompts.append(messages)
                yield AIMessageChunk(
                    content="代表菜品：北京烤鸭是北京代表性美食。[S1]\n\n"
                    "特色小吃：驴打滚是北京特色小吃。[S1]"
                )

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=question)
            return [event async for event in stream_events(graph, question, "local-food-search")]

        with (
            patch.dict(os.environ, {"AGENT_DEEPSEEK_FETCH_ENABLED": ""}),
            patch("backend.app.agent.service.create_chat_model", return_value=Model()),
            patch("backend.app.agent.tools.DDGS") as search,
        ):
            search.return_value.text.side_effect = [[{
                "href": link,
                "title": "北京代表性美食",
                "body": "北京烤鸭是北京代表性美食。驴打滚是北京特色小吃。",
            }]]
            events = asyncio.run(collect())

        self.assertEqual(search.return_value.text.call_count, 1, events)
        queries = [call.args[0] for call in search.return_value.text.call_args_list]
        self.assertEqual(queries, [question])
        search.return_value.text.assert_called_once_with(question, max_results=50)
        self.assertTrue(any(event["type"] == "tool_end" for event in events))
        answer = events[-1]["content"]
        self.assertIn("代表菜品", answer)
        self.assertIn("特色小吃", answer)
        self.assertIn(f"[来源 1]({link})", answer)
        self.assertIn("去哪体验", prompts[0][0].content)

    def test_semantic_search_uses_bounded_multi_query_plan_and_citations(self):
        from backend.app.intent.semantic import SemanticIntent

        question = "2026年11月14日 王者荣耀KPL年度总决赛的举办地点"
        link = "https://example.org/kpl-final-venue"
        venue_fact = "2026年11月14日王者荣耀KPL年度总决赛举办地为成都五粮液文化体育中心综合体育馆。"

        class Model:
            def bind_tools(self, _tools):
                return self

            async def ainvoke(self, _messages):
                return AIMessage(content='{"action":"answer"}')

            async def astream(self, _messages):
                yield AIMessageChunk(content=f"{venue_fact}[S1]")

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=question)
            return [event async for event in stream_events(graph, question, "event-venue-search")]

        with (
            patch.dict(os.environ, {"AGENT_DEEPSEEK_FETCH_ENABLED": ""}),
            patch("backend.app.agent.service.create_chat_model", return_value=Model()),
            patch("backend.app.agent.service.resolve_intent", new=AsyncMock(return_value=SemanticIntent(
                route="search",
                question=question,
                query="2026 KPL 11月14日 总决赛 场馆",
                queries=["2026 KPL 年度总决赛 11月14日 官方 场馆", "成都 KPL 总决赛 场馆 来源"],
            ))),
            patch("backend.app.agent.tools.DDGS") as search,
        ):
            search.return_value.text.return_value = [{
                "href": link, "title": "KPL年度总决赛场馆", "body": venue_fact,
            }]
            events = asyncio.run(collect())

        self.assertEqual(search.return_value.text.call_count, 3, events)
        queries = [call.args[0] for call in search.return_value.text.call_args_list]
        self.assertTrue(any("官方" in query for query in queries))
        self.assertTrue(any("成都 KPL 总决赛 场馆 来源" == query for query in queries))
        self.assertTrue(all(call.kwargs == {"max_results": 50} for call in search.return_value.text.call_args_list))
        self.assertIn(venue_fact, events[-1]["content"])
        self.assertIn(f"[来源 1]({link})", events[-1]["content"])

    def test_partial_evidence_review_does_not_replace_web_prose(self):
        from backend.app.intent.semantic import SemanticIntent

        question = "2026年11月14日 王者荣耀KPL年度总决赛的举办地点"
        link = "https://example.org/other-event-live"
        unrelated_excerpt = "10月30日KPL年度总决赛淘汰赛直播开始时间为18:00。"

        class Model:
            def bind_tools(self, _tools):
                return self

            async def ainvoke(self, _messages):
                return AIMessage(content='{"action":"partial","missing":"没有来源将11月14日赛事与场馆关联"}')

            async def astream(self, messages):
                if "检查能否" in messages[0].content:
                    yield AIMessageChunk(content='{"query":""}')
                else:
                    yield AIMessageChunk(content="赛事举办地为成都。[S1]\n\n11 月 14 日的具体场馆还没有明确来源。")

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=question)
            return [event async for event in stream_events(graph, question, "event-venue-unverified")]

        with (
            patch.dict(os.environ, {"AGENT_DEEPSEEK_FETCH_ENABLED": ""}),
            patch("backend.app.agent.service.create_chat_model", return_value=Model()),
            patch("backend.app.agent.service.resolve_intent", new=AsyncMock(return_value=SemanticIntent(
                route="search", question=question, query=question,
            ))),
            patch("backend.app.agent.tools.DDGS") as search,
        ):
            search.return_value.text.return_value = [{
                "href": link, "title": "其他日期赛事直播", "body": unrelated_excerpt,
            }]
            events = asyncio.run(collect())

        answer = events[-1]["content"]
        self.assertIn("赛事举办地为成都", answer)
        self.assertIn("11 月 14 日的具体场馆还没有明确来源", answer)
        self.assertNotIn(unrelated_excerpt, answer)

    def test_multi_stage_location_search_fills_venue_details_before_writing(self):
        import json
        from backend.app.intent.semantic import SemanticIntent

        question = "海岚艺术节在哪里举办"
        overview_url = "https://example.org/overview"
        organizer_url = "https://example.com/program"
        followup = "海岚艺术节 主办方 各阶段 场馆安排"
        draft = "闭幕演出在**宁港市文化中心**。[S1]"
        test = self
        checks = []

        class Model:
            async def ainvoke(self, messages):
                test.assertIn("各阶段和地点的对应关系", messages[0].content)
                data = json.loads(messages[-1].content)
                if len(data["已尝试"]) == 1:
                    test.assertIn("青石市和宁港市", data["结果"][0]["摘录"])
                    checks.append("overview")
                    return AIMessage(content=json.dumps({
                        "action": "search", "query": followup, "missing": "各阶段对应的场馆",
                    }))
                excerpt = data["结果"][0]["摘录"]
                test.assertIn("开幕展 | 青石市美术馆", excerpt)
                test.assertIn("闭幕演出 | 宁港市文化中心", excerpt)
                test.assertIn("艺术节现场可免费入场", excerpt)
                checks.append("venues")
                return AIMessage(content=json.dumps({
                    "action": "answer",
                    "table": {"title": "各阶段举办地点一览", "columns": ["阶段", "场馆"], "rows": [
                        {"cells": ["开幕展", "青石市美术馆"], "sources": [1]},
                        {"cells": ["闭幕演出", "宁港市文化中心"], "sources": [1]},
                    ]},
                    "details": [
                        {"label": "开演时间", "text": "闭幕演出从下午两点开始。", "sources": [1]},
                        {"label": "入场安排", "text": "艺术节现场可免费入场。", "sources": [1]},
                        {"label": "未取到的来源", "text": "不应显示这条。", "sources": [999]},
                    ],
                }))

            async def astream(self, messages):
                test.assertIn("即使问句很短", messages[0].content)
                test.assertIn("明细表会由服务端插入正文", messages[0].content)
                test.assertIn("下列补充事实也会由服务端放到正文末尾", messages[0].content)
                test.assertEqual(json.loads(messages[-1].content)[0]["编号"], "S1")
                checks.append("draft")
                yield AIMessageChunk(content=draft)

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=question)
            return [event async for event in stream_events(graph, question, "festival-venues")]

        with (
            patch.dict(os.environ, {"AGENT_DEEPSEEK_FETCH_ENABLED": ""}),
            patch("backend.app.agent.service.create_chat_model", return_value=Model()),
            patch("backend.app.agent.service.resolve_intent", new=AsyncMock(return_value=SemanticIntent(
                route="search", question=question, query="海岚艺术节 举办地",
            ))),
            patch("backend.app.agent.tools.DDGS") as search,
        ):
            search.return_value.text.side_effect = [
                [{"href": overview_url, "body": "艺术节在青石市和宁港市举办。"}],
                [{"href": organizer_url, "body": (
                    "海岚艺术节安排如下。\n阶段 | 场馆 | 时间\n--- | --- | ---\n"
                    "开幕展 | 青石市美术馆 | 上午十点\n闭幕演出 | 宁港市文化中心 | 下午两点\n"
                    "艺术节现场可免费入场。"
                )}],
            ]
            events = asyncio.run(collect())
            self.assertEqual([call.args[0] for call in search.return_value.text.call_args_list],
                             ["海岚艺术节 举办地", followup])

        self.assertEqual(checks, ["overview", "venues", "draft"])
        ref = f"[来源 1]({organizer_url})"
        answer = events[-1]["content"]
        self.assertTrue(answer.startswith(f"闭幕演出在**宁港市文化中心**。{ref}\n\n### 各阶段举办地点一览"))
        self.assertIn(f"| 开幕展 | 青石市美术馆 {ref} |", answer)
        self.assertIn(f"| 闭幕演出 | 宁港市文化中心 {ref} |", answer)
        self.assertIn(f"### 补充信息\n\n- **开演时间**：闭幕演出从下午两点开始。 {ref}", answer)
        self.assertTrue(answer.endswith(f"- **入场安排**：艺术节现场可免费入场。 {ref}"))
        self.assertNotIn("不应显示这条", answer)

    def test_local_food_search_respects_explicit_no_web_request(self):
        question = "查询北京有什么美食，不要联网"

        class Model:
            async def ainvoke(self, _messages):
                raise AssertionError("用户禁止联网时不应调用意图模型")

            async def astream(self, _messages):
                raise AssertionError("用户禁止联网时不应生成搜索答案")
                yield

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=question)
            return [event async for event in stream_events(graph, question, "local-food-no-web")]

        with (
            patch("backend.app.agent.service.create_chat_model", return_value=Model()),
            patch("backend.app.agent.tools.DDGS") as search,
        ):
            events = asyncio.run(collect())

        search.return_value.text.assert_not_called()
        self.assertFalse(any(event["type"] == "tool_start" for event in events))
        self.assertIn("你可以发给我已有的资料", events[-1]["content"])

    def test_evidence_review_can_supplement_partial_results_or_ask(self):
        import json

        question = "星河科技为什么裁员"
        source = "https://example.com/notice"

        class Model:
            def __init__(self, action):
                self.action = action

            async def ainvoke(self, messages):
                data = json.loads(messages[-1].content)
                if "当前请求" in data:
                    return AIMessage(content="invalid router output")
                if len(data["已尝试"]) == 1 and self.action == "clarify":
                    return AIMessage(content='{"action":"clarify","clarification":"你要核实的是哪个部门的调整？"}')
                return AIMessage(content='{"action":"search","query":"星河科技 人员调整公告"}')

            async def astream(self, messages):
                yield AIMessageChunk(content="据公告，人员调整已启动，具体原因尚未公开。[S1]")

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=question)
            return [frame async for frame in stream_events(graph, question, "review-evidence")]

        for action, expected_review, expected_calls in (
            ("search", ["search", "partial"], 2), ("clarify", ["clarify"], 1),
        ):
            with (
                self.subTest(action=action),
                patch("backend.app.agent.service.create_chat_model", return_value=Model(action)),
                patch("backend.app.agent.tools.DDGS") as ddgs,
            ):
                ddgs.return_value.text.return_value = [{"href": source, "body": "公司已启动人员调整，未披露原因。"}]
                events = asyncio.run(collect())
                self.assertEqual(ddgs.return_value.text.call_count, expected_calls)
                self.assertEqual([event["action"] for event in events if event["type"] == "evidence_review"], expected_review)
                if action == "clarify":
                    self.assertEqual(events[-1]["content"], "你要核实的是哪个部门的调整？")
                else:
                    self.assertIn(source, events[-1]["content"])
                    self.assertFalse(any(event["type"] == "quality" for event in events))

    def test_partial_public_answer_keeps_the_verified_fact_after_retry(self):
        import json

        question = "远山博物馆周二几点闭馆，门票多少钱"
        link = "https://example.org/hours"

        class Model:
            async def ainvoke(self, messages):
                payload = json.loads(messages[-1].content)
                if "当前请求" in payload:
                    return AIMessage(content=json.dumps({"route": "search", "question": question, "query": "远山博物馆 开放时间"}))
                return AIMessage(content='{"action":"search","query":"远山博物馆 门票","missing":"门票金额"}')

            async def astream(self, messages):
                if isinstance(messages[-1], ToolMessage):
                    yield AIMessageChunk(content="远山博物馆周二17:00闭馆。[S1]\n\n门票金额暂未查到。")

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=question)
            return [frame async for frame in stream_events(graph, question, "partial-public-answer")]

        with (
            patch.dict(os.environ, {"AGENT_DEEPSEEK_FETCH_ENABLED": ""}),
            patch("backend.app.agent.service.create_chat_model", return_value=Model()),
            patch("backend.app.agent.tools.DDGS") as search,
        ):
            search.return_value.text.side_effect = [[{"href": link, "body": "远山博物馆周二17:00闭馆。"}], []]
            frames = asyncio.run(collect())
            self.assertEqual(search.return_value.text.call_count, 2)
        self.assertIn("17:00闭馆", frames[-1]["content"])
        self.assertIn("门票金额暂未查到", frames[-1]["content"])

    def test_verification_followup_uses_only_an_explicit_event_in_the_same_thread(self):
        followup = "帮我查证"
        event = "星河科技为什么裁员"
        skills = load_skills(SKILLS_DIR)
        for text in (followup, "那就帮我联网核实一下吧", "请查证这条消息"):
            self.assertEqual(classify(text, skills)["status"], "clarify")
        self.assertNotEqual(classify("帮我查证星河科技裁员的消息", skills)["status"], "clarify")

        class Model:
            def bind_tools(self, _tools):
                return self

            async def astream(self, messages):
                yield AIMessageChunk(content="据公告，星河科技已启动人员调整。[S1]")

        async def exercise(search):
            graph = await build_agent(memory_db_path=None)

            async def collect(question, thread):
                return [frame async for frame in stream_events(graph, question, thread)]

            unknown = await collect(followup, "no-context")
            self.assertIn("你想查证哪件事", unknown[-1]["content"])
            search.assert_not_called()
            await collect(event, "event-context")
            known = await collect(followup, "event-context")
            self.assertTrue(any(frame.get("reason") == "继续核实上一轮明确的事件" for frame in known))
            self.assertIn("https://example.com/announcement", known[-1]["content"])
            isolated = await collect(followup, "another-thread")
            self.assertIn("你想查证哪件事", isolated[-1]["content"])
            self.assertEqual(search.call_count, 2)
            self.assertTrue(all("星河科技" in call.args[0] for call in search.call_args_list))
            await collect("谢谢", "event-context")
            after_greeting = await collect(followup, "event-context")
            self.assertIn("https://example.com/announcement", after_greeting[-1]["content"])
            await collect("远山科技为什么裁员", "event-context")
            before = search.call_count
            ambiguous = await collect(followup, "event-context")
            self.assertIn("不止一件事", ambiguous[-1]["content"])
            self.assertEqual(search.call_count, before)

        with (
            patch("backend.app.agent.service.create_chat_model", return_value=Model()),
            patch("backend.app.agent.tools.DDGS") as ddgs,
        ):
            search = ddgs.return_value.text
            search.return_value = [{"href": "https://example.com/announcement", "body": "星河科技已启动人员调整。"}]
            asyncio.run(exercise(search))

    def test_missing_evidence_can_requery_once_without_showing_rejected_draft(self):
        import json

        question = "星河科技为什么裁员"
        source = "https://example.com/announcement"
        test = self

        class Model:
            def bind_tools(self, tools):
                return self

            async def astream(self, messages):
                if "校验反馈" in messages[-1].content:
                    payload = json.loads(messages[-1].content)
                    test.assertEqual(len(payload["已尝试"]), 1)
                    yield AIMessageChunk(content='{"query":"星河科技 人员调整 公告"}')
                elif any(isinstance(message, ToolMessage) and message.status == "success" for message in messages):
                    yield AIMessageChunk(content="据公司公告，人员调整已经启动，公告未说明具体原因。[S1]")
                else:
                    yield AIMessageChunk(content="这是一段未经核验的草稿。")

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=question)
            return [event async for event in stream_events(graph, question, "evidence-retry")]

        with (
            patch("backend.app.agent.service.create_chat_model", return_value=Model()),
            patch("backend.app.agent.tools.DDGS") as search,
        ):
            for records in ([{"href": source, "body": "公司宣布启动人员调整，未披露原因。"}], []):
                search.return_value.text.reset_mock(side_effect=True)
                search.return_value.text.side_effect = [[], records]
                events = asyncio.run(collect())
                self.assertEqual(search.return_value.text.call_count, 2)
                self.assertEqual(sum(event["type"] == "evidence_retry" for event in events), 1)
                self.assertNotIn("未经核验的草稿", events[-1]["content"])
                self.assertEqual(source in events[-1]["content"], bool(records))

    def test_evaluates_supported_arithmetic(self):
        self.assertEqual(evaluate_expression("(17 + 3) * 2"), 40)

    def test_evidence_retry_rejects_equivalent_queries_and_urls(self):
        from backend.app.agent.evidence_retry import validated_query

        call = AIMessage(content="", tool_calls=[
            {"name": "web_search", "id": "first", "args": {"query": "星河科技 裁员"}},
        ])
        self.assertIsNone(validated_query('{"query":"星河科技，裁员"}', [call]))
        self.assertIsNone(validated_query('{"query":"https://example.com"}', [call]))
        self.assertEqual(validated_query('{"query":"星河科技 人员调整"}', [call]), "星河科技 人员调整")

    def test_concrete_event_checks_search_first_and_general_questions_can_answer_directly(self):
        test = self
        question = "黄仁勋为什么辞职"
        skills = load_skills(SKILLS_DIR)
        self.assertTrue(classify(question, skills)["requires_web_evidence"])
        self.assertTrue(classify("星河科技是不是已经裁员了", skills)["requires_web_evidence"])
        self.assertTrue(classify("人们为什么会辞职", skills)["general"])

        class FakeModel:
            def bind_tools(self, tools):
                test.assertEqual([tool.name for tool in tools], ["web_search"])
                return self

            async def astream(self, messages):
                if isinstance(messages[-1], ToolMessage):
                    yield AIMessageChunk(content="该事件已经证实。")
                else:
                    test.assertEqual(messages[-1].content, "人们为什么会辞职")
                    yield AIMessageChunk(content="常见原因包括职业发展和工作环境。")

        async def collect(text):
            graph = await build_agent(memory_db_path=None, initial_question=text)
            return [event async for event in stream_events(graph, text, "event-routing")]

        with (
            patch("backend.app.agent.service.create_chat_model", return_value=FakeModel()),
            patch("backend.app.agent.tools.DDGS") as search,
        ):
            search.return_value.text.return_value = []
            direct = asyncio.run(collect("人们为什么会辞职"))
            self.assertEqual(direct[-1]["content"], "常见原因包括职业发展和工作环境。")
            search.return_value.text.assert_not_called()
            checked = asyncio.run(collect(question))
            search.return_value.text.assert_called_once_with("黄仁勋 辞职 官方公告 核实", max_results=50)
            self.assertIn("暂时无法确认是否发生", checked[-1]["content"])
            self.assertNotIn("该事件已经证实", checked[-1]["content"])
            search.return_value.text.return_value = [
                {"href": "https://example.com/interview", "body": "黄仁勋出席了一场采访。"},
            ]
            checked = asyncio.run(collect(question))
            self.assertEqual(checked[-1]["content"], "该事件已经证实。")

    def test_rejects_code(self):
        with self.assertRaises(ValueError):
            evaluate_expression("__import__('os').system('whoami')")

    def test_translates_known_weather_code(self):
        self.assertEqual(weather_code_description(0), "晴朗")

    def test_mixed_tool_batch_returns_one_result_per_call_in_original_order(self):
        test = self

        class StrictModel:
            async def astream(self, messages):
                requested = next((msg for msg in messages if isinstance(msg, AIMessage) and msg.tool_calls), None)
                if requested is None:
                    yield AIMessageChunk(content="", tool_calls=[
                        {"id": "calc-1", "name": "calculator", "args": {"expression": "2+2"}},
                        {"id": "denied", "name": "get_weather", "args": {"location": "北京"}},
                        {"id": "calc-2", "name": "calculator", "args": {"expression": "3+3"}},
                    ])
                    return
                results = [msg for msg in messages if isinstance(msg, ToolMessage)]
                test.assertEqual([msg.tool_call_id for msg in results], [call["id"] for call in requested.tool_calls])
                test.assertEqual([msg.status for msg in results], ["success", "error", "success"])
                test.assertEqual((results[0].content, results[2].content), ("4", "6"))
                yield AIMessageChunk(content="两道算式的结果分别是 4 和 6。")

            def bind_tools(self, tools):
                test.assertNotIn("get_weather", [item.name for item in tools])
                return self

        model = StrictModel()

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question="2+2")
            return [event async for event in stream_events(graph, "2+2", "mixed-tool-batch")]

        with (
            patch.dict(os.environ, {"TOOL_RETRIEVAL_EMBEDDING_PROVIDER": ""}),
            patch("backend.app.agent.service.create_chat_model", return_value=model),
            patch("backend.app.agent.tools._get_json") as weather_request,
        ):
            events = asyncio.run(collect())
            weather_request.assert_not_called()

        self.assertEqual(events[-1], {"type": "done", "content": "两道算式的结果分别是 4 和 6。"})

    def test_tool_budget_allows_six_sequential_calls_and_stops_a_seventh(self):
        test = self

        class FakeModel:
            attempts = 0
            bindings = 0

            def bind_tools(self, tools):
                self.bindings += 1
                return self

            async def astream(self, messages):
                self.attempts += 1
                if self.attempts == 7:
                    test.assertIn("本轮工具额度已用完", messages[0].content)
                yield AIMessageChunk(content="", tool_calls=[
                    {"name": "calculator", "id": f"calc-{self.attempts}", "args": {"expression": "2+2"}},
                ])

        async def collect():
            graph = await build_agent(memory_db_path=None)
            return [frame async for frame in stream_events(graph, "2+2", "tool-quota")]

        model = FakeModel()
        with patch("backend.app.agent.service.create_chat_model", return_value=model):
            frames = asyncio.run(collect())
        self.assertEqual(model.attempts, 7)
        self.assertEqual(model.bindings, 6)
        self.assertEqual(sum(frame["type"] == "tool_end" for frame in frames), 6)
        self.assertEqual(frames[-1], {"type": "done", "content": "计算结果：4"})

    def test_rejects_unknown_model_provider(self):
        previous = __import__("os").environ.get("MODEL_PROVIDER")
        try:
            __import__("os").environ["MODEL_PROVIDER"] = "unknown"
            with self.assertRaises(ModelConfigurationError):
                get_model_settings()
        finally:
            if previous is None:
                __import__("os").environ.pop("MODEL_PROVIDER", None)
            else:
                __import__("os").environ["MODEL_PROVIDER"] = previous

    def test_skills_are_loaded_and_routed_by_explicit_trigger(self):
        skills = load_skills(SKILLS_DIR)
        matched = match_skills("帮我调研今天的人工智能新闻", skills)
        self.assertEqual([skill.name for skill in matched], ["联网调研"])
        self.assertEqual(match_skills("解释一段 Python 代码", skills), [])
        self.assertEqual(select_tool_names(matched, ["calculator", "web_search", "get_weather", "search_search_text"]), ["calculator", "web_search", "search_search_text"])
        candidates = select_top_tool_candidates("查询北京天气", ["get_weather", "search_news", "calendar_create"], {"get_weather": "查询当前天气", "search_news": "搜索新闻", "calendar_create": "创建日历"}, limit=1)
        self.assertEqual(candidates, ["get_weather"])

    def test_tool_retrieval_uses_local_lexical_fallback_without_embedding_configuration(self):
        tools = {
            "get_weather": SimpleNamespace(description="查询城市当前天气、气温和降水", args_schema=None),
            "calendar_create": SimpleNamespace(description="创建日历会议", args_schema=None),
        }
        names, mode = retrieve_tool_candidates("北京今天的天气", tools, limit=1)
        self.assertEqual(mode, "lexical")
        self.assertEqual(names, ["get_weather"])

    def test_intent_policy_clarifies_ambiguous_or_missing_context(self):
        skills = load_skills(SKILLS_DIR)
        self.assertEqual(classify("帮我查资料", skills)["status"], "allow")
        self.assertEqual(classify("帮我网上查人工智能动态", skills)["status"], "allow")
        self.assertEqual(classify("(17 + 3) * 2", skills)["status"], "allow")
        self.assertEqual(classify("帮我规划旅行", skills)["status"], "clarify")
