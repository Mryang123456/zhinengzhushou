"""内置工具、意图、技能和工具配额。"""

from types import SimpleNamespace
from unittest.mock import patch
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
                    self.assertTrue(any(event.get("passed") is True for event in events))

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
            search.return_value.text.assert_called_once_with("黄仁勋 辞职 官方公告 核实", max_results=5)
            self.assertIn("暂时无法确认是否发生", checked[-1]["content"])
            self.assertNotIn("该事件已经证实", checked[-1]["content"])
            search.return_value.text.return_value = [
                {"href": "https://example.com/interview", "body": "黄仁勋出席了一场采访。"},
            ]
            checked = asyncio.run(collect(question))
            self.assertIn("无法根据可引用资料确认", checked[-1]["content"])
            self.assertIn("https://example.com/interview", checked[-1]["content"])
            self.assertNotIn("该事件已经证实", checked[-1]["content"])

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
