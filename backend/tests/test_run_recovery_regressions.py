"""Prompt 迭代后的幂等恢复与阻塞执行失租回归。"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from backend.app.api.chat_run_store import ChatRunStore, SubmissionConflict
from backend.app.api.chat_run_worker import _execute
from backend.app.api.chat_run_reaper import run_chat_reaper
from backend.app.api.run_lease import RunLeaseConfig
from backend.tests.test_chat_run_leases import AsyncContext


class RunRecoveryRegressionTests(unittest.TestCase):
    def test_prompt_change_reuses_original_run_but_business_change_conflicts(self):
        async def exercise():
            run_id, message_id = uuid4(), uuid4()
            original = {"question": "原问题", "retry": False, "use_memory": True, "use_mcp": False,
                        "prompt_manifest": [{"version": "1"}], "prompt_manifest_hash": "old"}
            cursor = SimpleNamespace(fetchone=AsyncMock(side_effect=[None, (run_id, message_id, original)] * 5))
            conn = SimpleNamespace(execute=AsyncMock(return_value=cursor), transaction=Mock(return_value=AsyncContext()))
            store = ChatRunStore(SimpleNamespace(connection=Mock(return_value=AsyncContext(conn))))
            updated = {**original, "prompt_manifest": [{"version": "2"}], "prompt_manifest_hash": "new"}
            self.assertEqual(await store.submit("owner", "thread", message_id, message_id, updated), run_id)
            for field, value in (("question", "其他问题"), ("retry", True), ("use_memory", False), ("use_mcp", True)):
                with self.assertRaises(SubmissionConflict):
                    await store.submit("owner", "thread", message_id, message_id, {**updated, field: value})
            self.assertEqual(original["prompt_manifest_hash"], "old")
            self.assertFalse(any("UPDATE" in call.args[0] for call in conn.execute.await_args_list))

        asyncio.run(exercise())

    def test_lease_loss_interrupts_blocked_execution_and_skips_finish(self):
        async def exercise():
            blocked = asyncio.Event()
            cleaned = asyncio.Event()

            async def run_frames(*_):
                blocked.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()

            async def heartbeat(*_):
                await blocked.wait()
                return False

            store = SimpleNamespace(heartbeat=heartbeat, publish=AsyncMock(), finish=AsyncMock())
            with patch("backend.app.api.chat_run_worker.account_enabled", return_value=True), patch(
                "backend.app.api.chat_run_worker._run_frames", side_effect=run_frames,
            ):
                await asyncio.wait_for(_execute(
                    store, None, uuid4(), "owner", "thread", (uuid4(), {}, 1, False, uuid4()),
                    RunLeaseConfig(0.2, 0.01, 0.1),
                ), 0.5)
            self.assertTrue(cleaned.is_set())
            store.finish.assert_not_awaited()

        asyncio.run(exercise())

    def test_reaper_runs_without_available_execution_worker(self):
        async def exercise():
            reaped = asyncio.Event()

            async def reap(_):
                reaped.set()
                return [uuid4()]

            store = SimpleNamespace(
                pool=SimpleNamespace(connection=Mock(return_value=AsyncContext(object()))),
                reap_expired=AsyncMock(side_effect=reap),
            )
            task = asyncio.create_task(run_chat_reaper(store))
            try:
                await asyncio.wait_for(reaped.wait(), 0.5)
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            store.reap_expired.assert_awaited_once()

        asyncio.run(exercise())
