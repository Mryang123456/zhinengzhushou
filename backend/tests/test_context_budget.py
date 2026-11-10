"""模型上下文预算、完整回合窗口和摘要。"""

from unittest.mock import patch
import asyncio
import json
import os
import unittest

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from backend.app.agent.context import (
    context_budget,
    estimate_tokens,
    fallback_summary,
    messages_token_count,
    recent_window,
    split_turns,
)
from backend.app.agent.service import build_agent
from backend.app.agent.turns import checkpoint_config, stream_events


class ContextBudgetTests(unittest.TestCase):
    def test_context_window_keeps_complete_recent_tool_turns(self):
        messages = [
            HumanMessage("旧问题 " + "a" * 4_000),
            AIMessage("旧答案 " + "b" * 4_000),
            HumanMessage("查询天气"),
            AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"location": "北京"}, "id": "call-1", "type": "tool_call"}]),
            ToolMessage("北京晴朗", tool_call_id="call-1", name="get_weather"),
            AIMessage("北京晴朗。"),
        ]
        window = recent_window(messages, budget=400)
        self.assertEqual(len(split_turns(window)), 1)
        self.assertIsInstance(window[0], HumanMessage)
        self.assertTrue(any(isinstance(message, ToolMessage) for message in window))
        self.assertIn("查询天气", fallback_summary(messages))

    def test_context_window_does_not_force_an_oversized_previous_turn(self):
        question = HumanMessage("继续")
        history = [HumanMessage("早先的问题"), AIMessage("甲" * 6000), question]
        self.assertEqual(recent_window(history, 1024), [question])
        self.assertEqual(recent_window(history[:2], 1024), [])

    def test_model_and_summary_stay_within_budget_and_large_tool_output_stops_safely(self):
        from langchain_core.tools import tool
        from langchain_core.utils.function_calling import convert_to_openai_tool

        @tool
        def calculator(expression: str) -> str:
            """用于验证过大的工具结果。"""
            return large_result

        large_result = "证" * 9000
        test = self
        calls = []

        class FakeModel:
            schemas = []

            def bind_tools(self, tools):
                self.schemas = [convert_to_openai_tool(item) for item in tools]
                return self

            async def ainvoke(self, messages):
                calls.append("summary")
                test.assertLessEqual(messages_token_count(messages), context_budget())
                return AIMessage("摘要" * 3000)

            async def astream(self, messages):
                calls.append("answer")
                schema_tokens = estimate_tokens(json.dumps(self.schemas, ensure_ascii=False))
                test.assertLessEqual(messages_token_count(messages) + schema_tokens, context_budget())
                test.assertEqual(messages[-1].content, "2+2")
                yield AIMessageChunk(content="", tool_calls=[
                    {"name": "calculator", "id": "large", "args": {"expression": "2+2"}},
                ])

        async def collect():
            graph = await build_agent(memory_db_path=None)
            config = checkpoint_config("strict-budget")
            await graph.aupdate_state(config, {"messages": [
                HumanMessage("旧问题"), AIMessage("旧" * 12000),
            ]}, as_node="output")
            frames = [frame async for frame in stream_events(graph, "2+2", "strict-budget")]
            snapshot = await graph.aget_state(config)
            test.assertTrue(any(msg.content == large_result for msg in snapshot.values["messages"]))
            return frames

        with (
            patch.dict(os.environ, {"CONTEXT_WINDOW_TOKENS": "4096", "CONTEXT_RESPONSE_RESERVE_TOKENS": "2048", "TOOL_RETRIEVAL_EMBEDDING_PROVIDER": ""}),
            patch("backend.app.agent.service.TOOLS", [calculator]),
            patch("backend.app.agent.service.create_chat_model", return_value=FakeModel()),
        ):
            frames = asyncio.run(collect())
        self.assertEqual(calls, ["summary", "answer"])
        self.assertEqual(frames[-1]["type"], "done")
        self.assertIn("超过可处理的长度", frames[-1]["content"])

    def test_default_context_budget_is_128k_with_response_reserve(self):
        env = __import__("os").environ
        keys = ("CONTEXT_WINDOW_TOKENS", "CONTEXT_RESPONSE_RESERVE_TOKENS")
        previous = {key: env.pop(key, None) for key in keys}
        try:
            self.assertEqual(context_budget(), 131_072 - 8_192)
        finally:
            for key, value in previous.items():
                if value is not None:
                    env[key] = value
