"""会话检查点、同会话并发和重试。"""

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import asyncio
import json
import os
import tempfile
import unittest
import uuid

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from backend.app.agent.service import build_agent
from backend.app.agent.turn_gate import ThreadTurnGate, TurnQueueTimeout
from backend.app.agent.turns import checkpoint_config, checkpoint_scope, stream_events
from backend.app.memory.store import SQLiteMemoryStore
from backend.app.security import AuthenticatedUser


class TurnsTests(unittest.TestCase):
    def test_concurrent_http_turns_preserve_both_questions_in_checkpoint(self):
        from httpx import ASGITransport, AsyncClient
        from backend.app.api import server

        async def exercise():
            first_started = asyncio.Event()
            release_first = asyncio.Event()
            second_started = asyncio.Event()
            contexts = {}

            class FakeModel:
                async def astream(self, messages):
                    context = [msg.content for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]
                    question = context[-1]
                    contexts[question] = context
                    if question == "Question A":
                        first_started.set()
                        await release_first.wait()
                    if question == "Question B":
                        second_started.set()
                    yield AIMessageChunk(content=f"Reply to {question}")

            async with server.lifespan(server.app), AsyncClient(
                transport=ASGITransport(app=server.app), base_url="http://test",
            ) as client:
                async def post(question, thread):
                    response = await client.post("/chat/stream", json={
                        "question": question, "thread_id": thread, "message_id": str(uuid.uuid4()),
                    })
                    self.assertEqual(response.status_code, 200)
                    frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                    self.assertEqual(frames[-1]["type"], "done", frames)

                with patch("backend.app.agent.service.create_chat_model", return_value=FakeModel()):
                    async with asyncio.TaskGroup() as tasks:
                        tasks.create_task(post("Question A", "shared"))
                        await asyncio.wait_for(first_started.wait(), 3)
                        tasks.create_task(post("Question B", "shared"))
                        try:
                            await asyncio.wait_for(post("Another conversation", "independent"), 3)
                            self.assertFalse(second_started.is_set())
                        finally:
                            release_first.set()
                    checkpoint = await server.app.state.checkpointer.aget(checkpoint_config("shared", "parallel-user"))
                    messages = checkpoint["channel_values"]["messages"]
                    self.assertEqual([msg.content for msg in messages if isinstance(msg, HumanMessage)], ["Question A", "Question B"])
                    self.assertEqual(contexts["Question B"], ["Question A", "Reply to Question A", "Question B"])

        with (
            patch.dict(os.environ, {"CHECKPOINT_DATABASE_URL": ""}),
            patch.object(server, "verify_request"),
            patch.object(server, "authenticated_user", return_value=AuthenticatedUser("parallel-user")),
            patch.object(server, "ensure_model_ready"),
        ):
            asyncio.run(asyncio.wait_for(exercise(), 8))

    def test_thread_queue_timeout_does_not_block_later_turns(self):
        async def exercise():
            gate = ThreadTurnGate()
            async with gate.hold("same-thread", 1):
                with self.assertRaises(TurnQueueTimeout):
                    async with gate.hold("same-thread", 0.01):
                        self.fail("等待中的请求不能进入同一会话")
            async with gate.hold("same-thread", 1):
                pass

        asyncio.run(exercise())

    def test_postgres_turn_returns_connection_with_lock_and_unlocks_before_return(self):
        connection = SimpleNamespace(execute=AsyncMock(), close=AsyncMock())

        @asynccontextmanager
        async def checkout(*, timeout):
            yield connection
            self.assertEqual(connection.execute.await_count, 2)

        async def exercise():
            async with ThreadTurnGate().hold("user:thread", 1, SimpleNamespace(connection=checkout)) as locked:
                self.assertIs(locked, connection)
                self.assertEqual(connection.execute.await_count, 1)

        asyncio.run(exercise())
        acquired, released = connection.execute.await_args_list
        self.assertEqual(acquired.args[0], "SELECT pg_advisory_lock(%s::bigint)")
        self.assertEqual(released.args[0], "SELECT pg_advisory_unlock(%s::bigint)")
        self.assertEqual(acquired.args[1], released.args[1])
        connection.close.assert_not_called()

    def test_http_retry_reuses_checkpoint_question_and_replaces_old_answer(self):
        from fastapi.testclient import TestClient
        from backend.app.api import server

        thread_id = "retry-checkpoint"
        message_id = "83ca6850-31c9-40de-af8a-b55368608923"
        next_id = "2f0cad15-493f-48fc-b81e-c7c5301a3e42"
        checkpointer = MemorySaver()
        model_inputs = []

        class FakeModel:
            async def astream(self, messages):
                model_inputs.append(messages)
                yield AIMessageChunk(content=f"第 {len(model_inputs)} 次回答")

        @asynccontextmanager
        async def shared_checkpoint(_url):
            yield checkpointer

        def post(client, question_id=message_id, retry=False):
            response = client.post("/chat/stream", json={
                "question": "Python 的列表是什么？",
                "thread_id": thread_id,
                "message_id": question_id,
                "retry": retry,
            })
            self.assertEqual(response.status_code, 200)
            return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]

        with (
            patch.object(server, "verify_request"),
            patch.object(server, "authenticated_user", return_value=AuthenticatedUser("retry-user")),
            patch.object(server, "ensure_model_ready"),
            patch.object(server, "checkpoint_scope", shared_checkpoint),
            patch("backend.app.agent.service.create_chat_model", return_value=FakeModel()),
        ):
            client = TestClient(server.app)
            self.assertEqual(post(client)[-1]["type"], "done")
            self.assertEqual(post(client)[-1]["type"], "error")
            self.assertEqual(post(client, retry=True)[-1], {"type": "done", "content": "第 2 次回答"})
            self.assertEqual(post(client, retry=True)[-1], {"type": "done", "content": "第 3 次回答"})
            self.assertEqual(post(client, next_id)[-1]["type"], "done")
            self.assertEqual(post(client, retry=True)[-1]["retryable"], False)

        self.assertEqual([message.id for message in model_inputs[2] if isinstance(message, HumanMessage)], [f"web:{message_id}"])
        self.assertFalse(any(isinstance(message, AIMessage) for message in model_inputs[2]))
        checkpoint = checkpointer.get(checkpoint_config(thread_id, "retry-user"))
        history = checkpoint["channel_values"]["messages"]
        self.assertEqual([message.id for message in history if isinstance(message, HumanMessage)], [f"web:{message_id}", f"web:{next_id}"])
        self.assertEqual([message.content for message in history if isinstance(message, AIMessage)], ["第 3 次回答", "第 4 次回答"])

    def test_http_retry_keeps_read_tools_and_blocks_replayed_mutations(self):
        from fastapi.testclient import TestClient
        from langchain_core.tools import StructuredTool
        from backend.app.api import server
        from backend.app.agent import service

        async def search(query: str) -> str:
            """查询网页资料。"""
            return '{"href":"https://example.com/report","body":"资料包含示例。"}'

        write_call = AsyncMock()

        async def write_external(query: str) -> str:
            """写入外部系统。"""
            return await write_call(query)

        read = StructuredTool.from_function(coroutine=search, name="search_search_text")
        write = StructuredTool.from_function(coroutine=write_external, name="search_write")
        discovery = AsyncMock(return_value=[read, write])
        memory = SimpleNamespace(recent=lambda user: [(1, "偏好中文资料")], search=lambda user, query: [(1, "偏好中文资料")], remember=AsyncMock(), forget=AsyncMock())
        approval = SimpleNamespace(create=AsyncMock())
        checkpointer = MemorySaver()
        candidates = []

        class FakeModel:
            def bind_tools(self, tools):
                candidates.append({tool.name for tool in tools})
                return self

            async def astream(self, messages):
                if "本轮是重试" not in messages[0].content:
                    yield AIMessageChunk(content="首次回答")
                    return
                self_test.assertIn("偏好中文资料", messages[0].content)
                results = {msg.name: msg.status for msg in messages if isinstance(msg, ToolMessage)}
                if not results:
                    yield AIMessageChunk(content="", tool_calls=[
                        {"name": "search_search_text", "id": "search", "args": {"query": "资料"}},
                        {"name": "recall_memory", "id": "recall", "args": {"query": "偏好"}},
                        {"name": "remember_memory", "id": "remember", "args": {"content": "不应再次保存"}},
                        {"name": "forget_memory", "id": "forget", "args": {"memory_id": 1}},
                        {"name": "search_write", "id": "write", "args": {"query": "不应再次执行"}},
                    ])
                    return
                self_test.assertEqual(results, {
                    "search_search_text": "success", "recall_memory": "success",
                    "remember_memory": "error", "forget_memory": "error", "search_write": "error",
                })
                yield AIMessageChunk(content="资料包含示例 https://example.com/report")

        self_test = self
        governance = service.load_governance(server.MCP_CONFIG_FILE)
        governance["toolRisk"]["search_write"] = "L3"
        governance["maxToolBatch"] = 5

        @asynccontextmanager
        async def shared_checkpoint(_url):
            yield checkpointer

        payload = {"question": "帮我查资料", "thread_id": "retry-read-tools", "message_id": str(uuid.uuid4())}
        with (
            patch.dict(os.environ, {"KNOWLEDGE_DATABASE_URL": "", "AGENT_TOOL_CANDIDATE_LIMIT": "12", "TOOL_RETRIEVAL_EMBEDDING_PROVIDER": ""}),
            patch.object(server, "verify_request"),
            patch.object(server, "authenticated_user", return_value=AuthenticatedUser("test")),
            patch.object(server, "ensure_model_ready"),
            patch.object(server, "checkpoint_scope", shared_checkpoint),
            patch.object(service, "create_chat_model", return_value=FakeModel()),
            patch.object(service, "create_memory_store", return_value=memory),
            patch.object(service, "create_approval_store", return_value=approval),
            patch.object(service, "load_governance", return_value=governance),
            patch.object(service, "audit"),
            patch("backend.app.mcp.governance.audit"),
            patch("langchain_mcp_adapters.client.MultiServerMCPClient.get_tools", discovery),
        ):
            client = TestClient(server.app)
            for retry, expected in ((False, "首次回答"), (True, "资料包含示例 https://example.com/report")):
                response = client.post("/chat/stream", json={**payload, "retry": retry})
                self.assertEqual(response.status_code, 200)
                frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                self.assertEqual(frames[-1], {"type": "done", "content": expected})
                self.assertFalse(any(frame["type"] == "approval_required" for frame in frames))

        self.assertEqual(discovery.await_count, 2)
        self.assertTrue({"recall_memory", "search_search_text"}.issubset(candidates[-1]))
        self.assertFalse({"remember_memory", "forget_memory", "search_write"} & candidates[-1])
        for side_effect in (memory.remember, memory.forget, approval.create, write_call):
            side_effect.assert_not_called()

    def test_retry_replaces_legacy_question_and_unfinished_tool_chain(self):
        class FakeModel:
            async def astream(self, messages):
                self.messages = messages
                yield AIMessageChunk(content="新的回答")

        model = FakeModel()

        async def run():
            graph = await build_agent(memory_db_path=None)
            config = checkpoint_config("legacy-retry")
            await graph.aupdate_state(config, {"messages": [
                HumanMessage("更早的问题", id="earlier"),
                AIMessage("更早的回答", id="earlier-answer"),
                HumanMessage("解释元组", id="legacy-generated"),
                AIMessage(content="", id="old-tool-call", tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "call"}]),
                ToolMessage("2", tool_call_id="call", id="old-tool-result"),
            ]}, as_node="output")
            events = [event async for event in stream_events(graph, "解释元组", "legacy-retry", message_id="original", retry=True)]
            return events, (await graph.aget_state(config)).values["messages"]

        with (
            patch.dict(os.environ, {"KNOWLEDGE_DATABASE_URL": "", "TOOL_RETRIEVAL_EMBEDDING_PROVIDER": ""}),
            patch("backend.app.agent.service.create_chat_model", return_value=model),
        ):
            events, history = asyncio.run(run())
        self.assertEqual(events[-1], {"type": "done", "content": "新的回答"})
        self.assertEqual([message.id for message in history[:-1]], ["earlier", "earlier-answer", "web:original"])
        self.assertFalse(any(isinstance(message, ToolMessage) for message in model.messages))

    def test_new_question_recovers_from_incomplete_checkpoint_tool_batch(self):
        test = self

        class StrictModel:
            async def astream(self, messages):
                calls = [call["id"] for msg in messages if isinstance(msg, AIMessage) for call in msg.tool_calls]
                results = [msg.tool_call_id for msg in messages if isinstance(msg, ToolMessage)]
                test.assertEqual(calls, ["complete"])
                test.assertEqual(results, calls)
                test.assertEqual(messages[-1].content, "解释 Python 元组")
                yield AIMessageChunk(content="元组是不可变序列。")

        async def collect():
            graph = await build_agent(memory_db_path=None)
            config = checkpoint_config("new-after-disconnect")
            complete = {"name": "calculator", "id": "complete", "args": {"expression": "2+2"}}
            pending = {"name": "calculator", "id": "pending", "args": {"expression": "1+1"}}
            await graph.aupdate_state(config, {"messages": [
                HumanMessage("2+2"),
                AIMessage(content="", tool_calls=[complete]),
                ToolMessage("4", name="calculator", tool_call_id="complete"),
                HumanMessage("1+1"),
                AIMessage(content="", id="interrupted", tool_calls=[pending]),
            ]}, as_node="agent")
            frames = [frame async for frame in stream_events(graph, "解释 Python 元组", "new-after-disconnect", message_id="next")]
            snapshot = await graph.aget_state(config)
            test.assertNotIn("interrupted", [msg.id for msg in snapshot.values["messages"]])
            test.assertEqual(frames[-1], {"type": "done", "content": "元组是不可变序列。"})

        with patch("backend.app.agent.service.create_chat_model", return_value=StrictModel()):
            asyncio.run(collect())

    def test_checkpoint_key_is_user_isolated(self):
        self.assertEqual(
            checkpoint_config("会话-1", "张三")["configurable"]["thread_id"],
            "张三:会话-1",
        )
        self.assertNotEqual(checkpoint_config("会话-1", "张三"), checkpoint_config("会话-1", "李四"))

    def test_checkpoint_scope_uses_memory_saver_without_database(self):
        async def get_checkpointer():
            async with checkpoint_scope() as checkpointer:
                return checkpointer

        self.assertIsInstance(asyncio.run(get_checkpointer()), MemorySaver)

    def test_sqlite_memory_is_user_isolated_and_persistent(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "memory.db")
            memory_id = store.remember("张三", "我喜欢晴天出行。")
            self.assertEqual(store.recent("张三"), [(memory_id, "我喜欢晴天出行。")])
            self.assertEqual(store.recent("李四"), [])
            self.assertTrue(store.forget("张三", memory_id))
