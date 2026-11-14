"""使用专用测试库检查跨连接池恢复。"""

import asyncio
import os
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import AIMessageChunk, HumanMessage
from psycopg import AsyncConnection

from backend.app.agent.service import build_agent
from backend.app.agent.turn_gate import thread_lock_key
from backend.app.agent.turns import ask, checkpoint_config, checkpoint_scope


@unittest.skipUnless(os.getenv("TEST_CHECKPOINT_DATABASE_URL"), "未设置 PostgreSQL 专用测试库")
class PostgresCheckpointTests(unittest.TestCase):
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
