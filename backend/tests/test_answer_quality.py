"""证据校验、工具失败和被拒草稿隔离。"""

from unittest.mock import patch
import asyncio
import json
import os
import unittest

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import ToolException

from backend.app.agent.service import build_agent
from backend.app.agent.tools import calculator
from backend.app.agent.turns import checkpoint_config, stream_events
from backend.app.quality.validator import validate_answer


class AnswerQualityTests(unittest.TestCase):
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
            "数据为 42。结论可靠。https://example.com/report",
            "数据为 42 https://example.com/report https://example.com/other",
        ):
            with self.subTest(answer=answer):
                self.assertFalse(validate_answer("调研数据", answer, messages)["ok"])

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
