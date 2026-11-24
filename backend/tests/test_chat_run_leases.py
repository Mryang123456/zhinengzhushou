"""分布式聊天任务租约、心跳、回收和 fencing。"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

from backend.app.api.chat_run_store import ChatRunStore
from backend.app.api.run_lease import LeaseLost, RunLeaseConfig, maintain_run_lease


class AsyncContext:
    def __init__(self, value=None):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *_):
        return False


class ChatRunLeaseTests(unittest.TestCase):
    def test_lease_config_bounds_heartbeat_below_half_duration(self):
        with patch.dict("os.environ", {
            "AGENT_RUN_LEASE_SECONDS": "12",
            "AGENT_RUN_HEARTBEAT_SECONDS": "60",
            "AGENT_RUN_REAPER_SECONDS": "0",
        }):
            config = RunLeaseConfig.from_env()
        self.assertEqual(config.duration_seconds, 12)
        self.assertEqual(config.heartbeat_seconds, 6)
        self.assertEqual(config.reaper_seconds, 1)

    def test_heartbeat_marks_lease_lost_when_database_rejects_token(self):
        async def exercise():
            store = SimpleNamespace(heartbeat=AsyncMock(return_value=False))
            config = RunLeaseConfig(0.1, 0.01, 0.1)
            async with maintain_run_lease(store, uuid4(), uuid4(), config) as lost:
                await asyncio.wait_for(lost.wait(), 0.2)
            store.heartbeat.assert_awaited_once()

        asyncio.run(exercise())

    def test_blocked_heartbeat_cannot_outlive_local_lease_deadline(self):
        async def blocked_heartbeat(*_):
            await asyncio.sleep(1)
            return True

        async def exercise():
            store = SimpleNamespace(heartbeat=AsyncMock(side_effect=blocked_heartbeat))
            config = RunLeaseConfig(0.08, 0.01, 0.1)
            async with maintain_run_lease(store, uuid4(), uuid4(), config) as lost:
                await asyncio.wait_for(lost.wait(), 0.3)
            store.heartbeat.assert_awaited_once()

        asyncio.run(exercise())

    def test_claim_creates_unique_fencing_token_for_queued_run(self):
        async def exercise():
            returned = (uuid4(), {"question": "问题"}, 1, False, uuid4())
            cursor = SimpleNamespace(fetchone=AsyncMock(return_value=returned))
            conn = SimpleNamespace(execute=AsyncMock(return_value=cursor))
            claim = await ChatRunStore.claim(conn, uuid4(), "worker-a", 30)
            self.assertEqual(claim, returned)
            sql, args = conn.execute.await_args.args
            self.assertIn("status = 'queued'", sql)
            self.assertIn("lease_token", sql)
            self.assertEqual(args[0], "worker-a")
            self.assertEqual(args[2], 30)
            self.assertIsInstance(args[1], UUID)

        asyncio.run(exercise())

    def test_expired_lease_reaper_returns_requeued_runs(self):
        async def exercise():
            run_ids = [uuid4(), uuid4()]
            cursor = SimpleNamespace(fetchall=AsyncMock(return_value=[(value,) for value in run_ids]))
            conn = SimpleNamespace(execute=AsyncMock(return_value=cursor))
            self.assertEqual(await ChatRunStore.reap_expired(conn, 8), run_ids)
            sql, args = conn.execute.await_args.args
            self.assertIn("SKIP LOCKED", sql)
            self.assertIn("status = 'queued'", sql)
            self.assertEqual(args, (8,))

        asyncio.run(exercise())

    def test_stale_worker_cannot_publish_or_finish(self):
        async def exercise():
            cursor = SimpleNamespace(fetchone=AsyncMock(return_value=None))
            conn = SimpleNamespace(
                execute=AsyncMock(return_value=cursor),
                transaction=Mock(return_value=AsyncContext()),
            )
            pool = SimpleNamespace(connection=Mock(return_value=AsyncContext(conn)))
            store = ChatRunStore(pool)
            with self.assertRaises(LeaseLost):
                await store.publish(uuid4(), uuid4(), {"type": "progress"})
            with self.assertRaises(LeaseLost):
                await store.finish(conn, uuid4(), uuid4(), {"type": "done", "content": "不应提交"})
            self.assertEqual(conn.execute.await_count, 2)

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
