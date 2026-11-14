"""模型工具协议兼容与防泄漏测试。"""

import asyncio
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessageChunk, ToolMessage

from backend.app.agent.service import build_agent
from backend.app.agent.turns import stream_events
from backend.app.models.factory import ModelSettings, create_chat_model
from backend.app.models.tool_protocol import contains_textual_tool_protocol, parse_textual_tool_calls


class ModelToolProtocolTests(unittest.TestCase):
    def test_recovers_deepseek_dsml_as_allowlisted_tool_call(self):
        content = """<｜DSML｜calls>
<｜DSML｜invoke name="web_search">
<｜DSML｜parameter name="query" string="true">黄仁勋 最新消息</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜calls>"""

        calls = parse_textual_tool_calls(content, {"web_search"})

        self.assertIsNotNone(calls)
        self.assertEqual(calls[0]["name"], "web_search")
        self.assertEqual(calls[0]["args"], {"query": "黄仁勋 最新消息"})

    def test_recovers_rendered_double_pipe_variant(self):
        content = """< || DSML || calls>
< || DSML || invoke name="get_weather">
< || DSML || parameter name="location" string="true">杭州</ || DSML || parameter>
</ || DSML || invoke>
</ || DSML || calls>"""

        calls = parse_textual_tool_calls(content, {"get_weather"})

        self.assertEqual(calls[0]["args"], {"location": "杭州"})

    def test_rejects_unknown_or_malformed_protocol_without_hiding_detection(self):
        unknown = '<｜DSML｜calls><｜DSML｜invoke name="delete_all"></｜DSML｜invoke></｜DSML｜calls>'
        mixed = '请执行：<｜DSML｜calls></｜DSML｜calls>'

        self.assertTrue(contains_textual_tool_protocol(unknown))
        self.assertIsNone(parse_textual_tool_calls(unknown, {"web_search"}))
        self.assertIsNone(parse_textual_tool_calls(mixed, {"web_search"}))

    def test_deepseek_uses_non_thinking_mode_for_standard_tool_calls(self):
        settings = ModelSettings(
            provider="openai",
            name="deepseek-flash",
            temperature=0,
            base_url="https://api.deepseek.com",
        )
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("langchain_openai.ChatOpenAI") as constructor,
        ):
            create_chat_model(settings)

        self.assertEqual(
            constructor.call_args.kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    def test_agent_executes_recovered_protocol_instead_of_showing_it(self):
        class TextProtocolModel:
            def bind_tools(self, _tools):
                return self

            async def astream(self, messages):
                if any(isinstance(message, ToolMessage) for message in messages):
                    yield AIMessageChunk(content="计算结果：40")
                    return
                yield AIMessageChunk(content=(
                    '<｜DSML｜calls><｜DSML｜invoke name="calculator">'
                    '<｜DSML｜parameter name="expression" string="true">(17 + 3) * 2'
                    '</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜calls>'
                ))

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question="计算 (17 + 3) * 2")
            return [event async for event in stream_events(graph, "计算 (17 + 3) * 2", "dsml-recovery")]

        with patch("backend.app.agent.service.create_chat_model", return_value=TextProtocolModel()):
            events = asyncio.run(collect())

        self.assertTrue(any(event["type"] == "tool_protocol_recovered" for event in events))
        self.assertEqual(events[-1], {"type": "done", "content": "计算结果：40"})


if __name__ == "__main__":
    unittest.main()
