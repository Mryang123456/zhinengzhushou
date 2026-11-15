"""使用专用测试库检查跨连接池恢复。"""

import asyncio
import os
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from backend.app.agent.service import build_agent
from backend.app.agent.turn_gate import thread_lock_key
from backend.app.agent.turns import ask, checkpoint_config, checkpoint_scope, stream_events
from backend.app.api.chat_history import serialize_chat_history
from backend.app.api.chat_run_store import ChatRunStore


@unittest.skipUnless(os.getenv("TEST_CHECKPOINT_DATABASE_URL"), "未设置 PostgreSQL 专用测试库")
class PostgresCheckpointTests(unittest.TestCase):
    def test_cross_pool_resume_replays_final_answer_and_public_history(self):
        database = os.environ["TEST_CHECKPOINT_DATABASE_URL"]
        thread, owner = uuid4().hex, uuid4().hex
        message_id = uuid4()
        config = checkpoint_config(thread, owner)
        sources = [{"url": "https://example.org/report", "domain": "example.org", "title": "报告"}]

        async def exercise():
            async with AsyncConnectionPool(database, open=False, kwargs={"autocommit": True}) as pool:
                store = ChatRunStore(pool)
                await store.setup()
                run_id = await store.submit(owner, thread, message_id, message_id, {"question": "查报告"})
                try:
                    async with pool.connection() as conn:
                        await store.claim(conn, run_id)
                    await store.publish(run_id, 1, {"type": "tool_end", "tools": ["web_search"], "web_result_count": 2})
                    await store.publish(run_id, 1, {"type": "approval_required", "confirmation_token": "不应进入历史"})
                    before = await store.events(owner, run_id, 0)
                    async with checkpoint_scope(database) as saver:
                        graph = await build_agent(memory_db_path=None, checkpointer=saver)
                        await graph.aupdate_state(config, {
                            "messages": [HumanMessage("查报告", id=f"web:{message_id}"), AIMessage("核验后的回答")],
                            "durable_run_id": str(run_id),
                        }, as_node="quality")

                    async with checkpoint_scope(database) as saver:
                        graph = await build_agent(memory_db_path=None, checkpointer=saver, recovering=True)
                        frames = [item async for item in stream_events(
                            graph, "查报告", thread, owner, message_id=str(message_id), run_id=str(run_id),
                        )]
                        self.assertEqual(sum(isinstance(item, HumanMessage) for item in (await graph.aget_state(config)).values["messages"]), 1)
                        await saver.adelete_thread(config["configurable"]["thread_id"])

                    async with pool.connection() as conn:
                        await store.finish(conn, run_id, {**frames[-1], "sources": sources}, frames[:-1])
                    replay = await store.events(owner, run_id, before[-1][1])
                    self.assertEqual([event[2]["type"] for event in replay], ["token", "done"])
                    history = serialize_chat_history(await store.recent_sessions(owner))
                    reply = history[0]["messages"][-1]
                    self.assertEqual(reply["content"], "核验后的回答")
                    self.assertEqual(reply["sources"], sources)
                    self.assertGreaterEqual(reply["elapsedMs"], 0)
                    self.assertEqual(reply["activity"], [{"type": "tool_end", "tools": ["web_search"], "web_result_count": 2}])
                    self.assertEqual(await store.recent_sessions("another-owner"), [])
                finally:
                    async with pool.connection() as conn:
                        await conn.execute("DELETE FROM chat_runs WHERE id = %s", (run_id,))

        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
            runner.run(exercise())

    def test_second_connection_can_take_over_after_lock_owner_disconnects(self):
        async def exercise():
            database = os.environ["TEST_CHECKPOINT_DATABASE_URL"]
            key = thread_lock_key(uuid4().hex)
            sql = "SELECT pg_try_advisory_lock(%s::bigint)"
            async with await AsyncConnection.connect(database, autocommit=True) as second:
                async with await AsyncConnection.connect(database, autocommit=True) as first:
                    self.assertTrue((await (await first.execute(sql, (key,))).fetchone())[0])
                    self.assertFalse((await (await second.execute(sql, (key,))).fetchone())[0])
                self.assertTrue((await (await second.execute(sql, (key,))).fetchone())[0])

        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
            runner.run(exercise())

    def test_restores_context_after_reopening_pool_and_isolates_users(self):
        thread = uuid4().hex
        owner, other = "checkpoint-owner", "checkpoint-other"
        first = "我的出游地点是杭州"
        followup = "刚才提到哪里？"
        database = os.environ["TEST_CHECKPOINT_DATABASE_URL"]

        class Model:
            async def astream(self, messages):
                user_messages = [message.content for message in messages if isinstance(message, HumanMessage)]
                yield AIMessageChunk(content="地点是杭州。" if first in user_messages else "没有地点信息。")

        async def exercise():
            try:
                async with checkpoint_scope(database) as saver:
                    graph = await build_agent(memory_db_path=None, checkpointer=saver)
                    await ask(graph, first, thread, owner)
                async with checkpoint_scope(database) as saver:
                    graph = await build_agent(memory_db_path=None, checkpointer=saver)
                    self.assertEqual(await ask(graph, followup, thread, owner), "地点是杭州。")
                    self.assertEqual(await ask(graph, followup, thread, other), "没有地点信息。")
            finally:
                async with checkpoint_scope(database) as saver:
                    for user in (owner, other):
                        await saver.adelete_thread(checkpoint_config(thread, user)["configurable"]["thread_id"])

        loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
        with patch("backend.app.agent.service.create_chat_model", return_value=Model()):
            with asyncio.Runner(loop_factory=loop_factory) as runner:
                runner.run(exercise())
