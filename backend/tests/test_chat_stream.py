"""聊天流式事件和 HTTP 入口。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import asyncio
import json
import os
import unittest

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from backend.app.agent.service import build_agent
from backend.app.agent.turns import stream_events
from backend.app.observability import observe_duration, prometheus_metrics
from backend.app.security import AuthenticatedUser


class ChatStreamTests(unittest.TestCase):
    def test_stream_events_include_token_tool_and_done(self):
        class FakeGraph:
            async def astream(self, *_args, **_kwargs):
                # agent 草稿不应由 stream_events 直接转发；仅 output 节点写入的 custom token 可见。
                yield "updates", {"tools": {"messages": [SimpleNamespace(name="天气工具")]}}
                yield "updates", {"agent": {"messages": [SimpleNamespace(content="未核验草稿")]}}
                yield "custom", {"type": "token", "content": "已核验答案"}

        async def collect():
            return [event async for event in stream_events(FakeGraph(), "测试", "thread-1")]

        events = asyncio.run(collect())
        self.assertEqual([event["type"] for event in events], ["tool_end", "token", "done"])
        self.assertEqual(events[1]["content"], "已核验答案")
        self.assertEqual(events[-1]["content"], "已核验答案")

    def test_real_graph_stream_finishes_when_nodes_return_no_update(self):
        class FakeModel:
            def bind_tools(self, _tools):
                return self

            async def astream(self, _messages):
                yield AIMessageChunk(content="计算结果是 40。")

        async def collect():
            graph = await build_agent(memory_db_path=None)
            return [event async for event in stream_events(graph, "(17 + 3) * 2", "stream-regression")]

        with (
            patch.dict(os.environ, {"KNOWLEDGE_DATABASE_URL": "", "TOOL_RETRIEVAL_EMBEDDING_PROVIDER": ""}),
            patch("backend.app.agent.service.create_approval_store"),
            patch("backend.app.agent.service.create_chat_model", return_value=FakeModel()),
        ):
            events = asyncio.run(collect())

        self.assertEqual(events[-1], {"type": "done", "content": "计算结果是 40。"})
        self.assertEqual([event["content"] for event in events if event["type"] == "token"], ["计算结果是 40。"])

    def test_chat_http_finishes_with_default_mcp_enabled(self):
        from fastapi.testclient import TestClient
        from backend.app.api import server

        answer = "Python 的列表是一种有序容器。" * 8

        class FakeModel:
            bound_tools = ()

            def bind_tools(self, tools):
                self.bound_tools = tuple(tool.name for tool in tools)
                return self

            async def astream(self, _messages):
                yield AIMessageChunk(content=answer[:50])
                yield AIMessageChunk(content=answer[50:])

        for question, expects_mcp in (
            ("Python 的列表是什么？", False),
            ("帮我查资料", True),
        ):
            with self.subTest(question=question):
                model = FakeModel()
                discovery = AsyncMock(side_effect=OSError("MCP unavailable"))
                with (
                    patch.dict(os.environ, {
                        "CHECKPOINT_DATABASE_URL": "", "MEMORY_DATABASE_URL": "",
                        "KNOWLEDGE_DATABASE_URL": "", "TOOL_RETRIEVAL_EMBEDDING_PROVIDER": "",
                    }),
                    patch.object(server, "verify_request"),
                    patch.object(server, "authenticated_user", return_value=AuthenticatedUser("test")),
                    patch.object(server, "ensure_model_ready"),
                    patch("backend.app.agent.service.create_memory_store", return_value=None),
                    patch("backend.app.agent.service.create_chat_model", return_value=model),
                    patch("langchain_mcp_adapters.client.MultiServerMCPClient.get_tools", discovery),
                ):
                    response = TestClient(server.app).post("/chat/stream", json={"question": question})
                self.assertEqual(response.status_code, 200)
                frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                self.assertFalse(any(frame["type"] == "error" for frame in frames), frames)
                chunks = [frame["content"] for frame in frames if frame["type"] == "token"]
                self.assertGreater(len(chunks), 1)
                self.assertEqual("".join(chunks), answer)
                self.assertEqual(frames[-1], {"type": "done", "content": answer})
                if expects_mcp:
                    discovery.assert_awaited_once()
                    self.assertIn("web_search", model.bound_tools)
                else:
                    discovery.assert_not_awaited()
                    self.assertEqual(model.bound_tools, ())

    def test_local_web_checkpoint_preserves_turns_and_isolates_threads(self):
        from fastapi.testclient import TestClient
        from backend.app.api import server

        model_inputs = []

        class FakeModel:
            async def astream(self, messages):
                model_inputs.append([message.content for message in messages if isinstance(message, (HumanMessage, AIMessage))])
                yield AIMessageChunk(content=f"answer {len(model_inputs)}")

        def ask(client, question, thread_id, user="user-a"):
            response = client.post(
                "/chat/stream", json={"question": question, "thread_id": thread_id},
                headers={"Authorization": user},
            )
            self.assertEqual(response.status_code, 200)
            frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            self.assertEqual(frames[-1]["type"], "done", frames)

        with (
            patch.dict(os.environ, {"CHECKPOINT_DATABASE_URL": ""}),
            patch.object(server, "verify_request"),
            patch.object(server, "authenticated_user", side_effect=lambda user: AuthenticatedUser(user)),
            patch.object(server, "ensure_model_ready"),
            patch("backend.app.agent.service.create_chat_model", return_value=FakeModel()),
        ):
            with TestClient(server.app) as client:
                ask(client, "Remember the context for this explanation", "thread-a")
                ask(client, "Continue with an example", "thread-a")
                ask(client, "A separate conversation", "thread-b")
                ask(client, "A different user", "thread-a", user="user-b")
            with TestClient(server.app) as client:
                ask(client, "After a server restart", "thread-a")

        self.assertEqual(model_inputs, [
            ["Remember the context for this explanation"],
            ["Remember the context for this explanation", "answer 1", "Continue with an example"],
            ["A separate conversation"],
            ["A different user"],
            ["After a server restart"],
        ])

    def test_prometheus_exposes_duration_series(self):
        observe_duration("unit_test", 0.25)
        metrics = prometheus_metrics()
        self.assertIn('agent_operation_duration_seconds_sum{operation="unit_test"}', metrics)
