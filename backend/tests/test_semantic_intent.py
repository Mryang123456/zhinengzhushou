"""语义追问、会话隔离与外部检索约束。"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from backend.app.agent.service import build_agent
from backend.app.agent.turns import stream_events
from backend.app.intent.capabilities import describe_capabilities
from backend.app.intent.policy import classify


class Model:
    def __init__(self):
        self.revision_question = ""
        self.router_inputs = []

    async def ainvoke(self, messages):
        data = json.loads(messages[-1].content)
        self.router_inputs.append(data)
        question = data["当前请求"]
        history = data["近期对话"]
        if "最近财报数字" in question and history:
            result = {"route": "search", "question": "查询星河科技最近财报数字", "query": "星河科技 官方财报"}
        elif "最近财报数字" in question:
            result = {"route": "clarify", "question": question, "clarification": "你想查哪家公司的财报？"}
        else:
            result = {"route": "direct", "question": question}
        return AIMessage(content=json.dumps(result, ensure_ascii=False))

    def bind_tools(self, _tools):
        return self

    async def astream(self, messages):
        latest = messages[-1]
        if "上一版校验原因" in latest.content:
            self.revision_question = latest.content
            yield AIMessageChunk(content="报告披露营收为 20 亿元。[S1]")
        elif isinstance(latest, ToolMessage):
            yield AIMessageChunk(content="草稿没有引用。")
        else:
            yield AIMessageChunk(content="我们来聊星河科技。")


class SemanticIntentTests(unittest.TestCase):
    def test_weather_clothing_uses_weather_tool_even_if_model_selects_public_search(self):
        question = "上海现在适合穿什么衣服"
        report = "上海当前天气：晴朗；气温 12°C，体感 10°C。"
        self.assertTrue(classify(question, [])["weather_clothing"])
        self.assertTrue(classify("明天杭州怎么穿", [])["weather_clothing"])
        self.assertFalse(classify("参加面试穿什么衣服", []).get("weather_assessment"))
        self.assertFalse(classify("气温是 15℃ 时怎么穿", []).get("weather_assessment"))

        class ClothingModel(Model):
            async def ainvoke(self, messages):
                return AIMessage(content=json.dumps({
                    "route": "search", "question": "上海现在（2026年11月14日）请依据当前天气和气温给出穿衣建议",
                    "query": "上海今日穿衣指数",
                }, ensure_ascii=False))

            def bind_tools(self, tools):
                assert [tool.name for tool in tools] == ["get_weather"]
                return self

            async def astream(self, messages):
                if messages[-1].content != question:
                    data = json.loads(messages[-1].content)
                    assert data["已核验预报"] == report
                    yield AIMessageChunk(content='{"activity":"","clothing":"建议穿长袖，外搭一件挡风外套。"}')
                else:
                    yield AIMessageChunk(content="", tool_calls=[
                        {"name": "get_weather", "id": "weather", "args": {"location": "上海", "date": "current"}},
                    ])

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=question)
            return [frame async for frame in stream_events(graph, question, "clothing")]

        with (
            patch("backend.app.agent.service.create_chat_model", return_value=ClothingModel()),
            patch("backend.app.agent.tools.query_weather", return_value=report) as weather,
            patch("backend.app.agent.tools.DDGS") as ddgs,
        ):
            events = asyncio.run(collect())
            self.assertEqual(weather.call_args.args[:3], ("上海", "current", 1))
            ddgs.assert_not_called()
        self.assertIn("挡风外套", events[-1]["content"])
        self.assertTrue(events[-1]["content"].endswith(report))

    def test_rewrite_reaches_search_and_revision_and_does_not_cross_threads(self):
        model = Model()

        async def collect():
            graph = await build_agent(memory_db_path=None)

            async def turn(question, thread):
                return [frame async for frame in stream_events(graph, question, thread)]

            await turn("介绍一下星河科技", "same")
            return await turn("最近财报数字", "same"), await turn("最近财报数字", "other")

        with (
            patch("backend.app.agent.service.create_chat_model", return_value=model),
            patch("backend.app.agent.tools.DDGS") as ddgs,
        ):
            ddgs.return_value.text.return_value = [
                {"href": "https://example.com/report", "body": "报告披露营收为 20 亿元。"},
            ]
            resolved, missing = asyncio.run(collect())
            ddgs.return_value.text.assert_called_once_with("星河科技 官方财报", max_results=5)
        self.assertIn("查询星河科技最近财报数字", model.revision_question)
        self.assertIn("20 亿元", resolved[-1]["content"])
        capabilities = model.router_inputs[1]["后端能力"]
        self.assertTrue(capabilities["public_search"])
        self.assertIn("get_weather", {tool["name"] for tool in capabilities["tools"]})
        self.assertFalse(capabilities["knowledge_base"]["available_this_turn"])
        self.assertGreater(capabilities["remaining_tool_calls"], 0)
        self.assertIn("哪家公司", missing[-1]["content"])
        self.assertFalse(any(frame["type"] == "tool_start" for frame in missing))

    def test_no_web_instruction_blocks_semantic_search(self):
        model = Model()
        async def collect():
            graph = await build_agent(memory_db_path=None)
            async for _ in stream_events(graph, "介绍一下星河科技", "same"):
                pass
            return [frame async for frame in stream_events(graph, "最近财报数字，不要 联网", "same")]

        with (
            patch("backend.app.agent.service.create_chat_model", return_value=model),
            patch("backend.app.agent.tools.DDGS") as ddgs,
        ):
            events = asyncio.run(collect())
            ddgs.assert_not_called()
        self.assertFalse(any(frame["type"] == "tool_start" for frame in events))
        self.assertFalse(model.router_inputs[-1]["后端能力"]["public_search"])

    def test_capabilities_filter_retry_and_report_approval_and_budget(self):
        tool = SimpleNamespace(description="本地工具")
        mcp = SimpleNamespace(description="不应进入路由请求的外部说明")
        registry = {"web_search": tool, "search_knowledge_base": tool, "remember_memory": tool, "calendar_create": mcp}
        options = {
            "mcp_names": {"calendar_create"}, "governance": {}, "blocked_tools": {"remember_memory"},
            "max_total": 2, "max_batch": 3, "public_search_allowed": True, "knowledge_configured": True,
        }
        summary = describe_capabilities(registry, **options)
        tools = {item["name"]: item for item in summary["tools"]}
        self.assertNotIn("remember_memory", tools)
        self.assertEqual(tools["calendar_create"]["access"], "requires_approval")
        self.assertEqual(tools["calendar_create"]["description"], "")
        self.assertEqual(summary["max_calls_per_batch"], 2)
        self.assertTrue(summary["knowledge_base"]["requires_explicit_request"])
        self.assertTrue(summary["knowledge_base"]["available_this_turn"])
        disabled = describe_capabilities(registry, **(options | {"max_total": 0}))
        self.assertEqual(disabled["tools"], [])
        self.assertFalse(disabled["public_search"])
        self.assertTrue(disabled["knowledge_base"]["registered"])
        self.assertFalse(disabled["knowledge_base"]["available_this_turn"])
