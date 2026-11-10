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
    def test_evaluates_supported_arithmetic(self):
        self.assertEqual(evaluate_expression("(17 + 3) * 2"), 40)

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
