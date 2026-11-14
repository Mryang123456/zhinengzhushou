"""持久化聊天任务的检查点恢复与事件读取权限。"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from backend.app.agent.service import build_agent
from backend.app.agent.turns import checkpoint_config, stream_events
from backend.app.api import server
from backend.app.security import AuthenticatedUser


class ChatRunTests(unittest.TestCase):
    def test_session_history_only_reads_authenticated_account_and_returns_final_text(self):
        from datetime import datetime, timezone

        message_id = uuid4()
        store = SimpleNamespace(recent_sessions=AsyncMock(return_value=[
            ("test-thread", message_id, "杭州天气", datetime.now(timezone.utc), "已有预报。"),
        ]))
        with (
            patch.object(server.app.state, "chat_runs", store, create=True),
            patch.object(server, "service_key_valid", return_value=True),
            patch.object(server, "authenticated_user", return_value=AuthenticatedUser("owner")),
            patch.object(server, "auth_required", return_value=True),
        ):
            response = TestClient(server.app).get("/chat/sessions?user_id=another")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertEqual(response.json()["sessions"][0]["messages"], [
                {"id": str(message_id), "role": "user", "content": "杭州天气"},
                {"id": f"{message_id}:answer", "role": "assistant", "content": "已有预报。"},
            ])
            store.recent_sessions.assert_awaited_once_with("owner")

    def test_submission_recovery_reuses_initial_and_retry_request_keys(self):
        run_id, message_id, attempt_id = uuid4(), uuid4(), uuid4()
        store = SimpleNamespace(
            submit=AsyncMock(return_value=run_id),
            events=AsyncMock(return_value=[("completed", 19, {"type": "done", "content": "已完成"})]),
        )
        with (
            patch.object(server.app.state, "chat_runs", store, create=True),
            patch.object(server, "verify_request"),
            patch.object(server, "authenticated_user", return_value=AuthenticatedUser("owner")),
            patch.object(server, "ensure_model_ready"),
        ):
            client = TestClient(server.app)
            initial = {"thread_id": "thread", "message_id": str(message_id), "question": "原问题"}
            retry = {**initial, "retry": True, "attempt_id": str(attempt_id)}
            for payload in (initial, retry):
                for recovery in (False, True):
                    response = client.post("/chat/stream", json={**payload, "recover_submission": recovery})
                    self.assertEqual(response.status_code, 200)
                    self.assertIn(str(run_id), response.text)
            self.assertEqual(client.post("/chat/stream", json={
                **initial, "retry": True, "recover_submission": True,
            }).status_code, 422)
        calls = [call.args for call in store.submit.await_args_list]
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[2], calls[3])
        self.assertEqual(calls[0][3], message_id)
        self.assertEqual(calls[2][3], attempt_id)

    def test_approval_recovery_requires_owned_run_and_consumes_existing_token(self):
        run_id = uuid4()
        token = "persisted-test-confirmation"
        store = SimpleNamespace(approval_token=AsyncMock(side_effect=lambda user, *_: token if user == "owner" else None))
        approvals = SimpleNamespace(consume=Mock(return_value=None))
        with (
            patch.object(server.app.state, "chat_runs", store, create=True),
            patch.object(server, "approval_store", approvals),
            patch.object(server, "verify_request"),
            patch.object(server, "authenticated_user", side_effect=lambda authorization: AuthenticatedUser(authorization)),
        ):
            client = TestClient(server.app)
            url = "/mcp/approvals/approval-test/confirm"
            for user in ("other", "owner"):
                response = client.post(url, json={"run_id": str(run_id)}, headers={"Authorization": user})
                self.assertEqual(response.status_code, 409)
            self.assertNotIn(token, response.text)
        store.approval_token.assert_any_await("owner", run_id, "approval-test")
        approvals.consume.assert_called_once_with("approval-test", "owner", token)

    def test_recovery_uses_pending_checkpoint_and_completed_snapshot(self):
        async def exercise():
            run_id = str(uuid4())
            message_id = str(uuid4())
            config = checkpoint_config("thread", "owner")
            model = SimpleNamespace(astream=AsyncMock())
            with patch("backend.app.agent.service.create_chat_model", return_value=model):
                graph = await build_agent(memory_db_path=None, recovering=True)
                await graph.aupdate_state(config, {
                    "messages": [
                        HumanMessage("原问题", id=f"web:{message_id}"),
                        AIMessage("已校验的回答"),
                    ],
                    "durable_run_id": run_id,
                }, as_node="quality")
                first = [event async for event in stream_events(
                    graph, "原问题", "thread", "owner", message_id=message_id, run_id=run_id,
                )]
                second = [event async for event in stream_events(
                    graph, "原问题", "thread", "owner", message_id=message_id, run_id=run_id,
                )]
            done = {"type": "done", "content": "已校验的回答"}
            self.assertEqual(first[-1], done)
            self.assertEqual(second, [done])
            model.astream.assert_not_called()
            snapshot = await graph.aget_state(config)
            self.assertEqual(sum(isinstance(item, HumanMessage) for item in snapshot.values["messages"]), 1)

        asyncio.run(exercise())

    def test_event_replay_and_cancel_are_scoped_to_authenticated_owner(self):
        run_id = uuid4()
        store = SimpleNamespace(
            status=AsyncMock(side_effect=lambda user, _: "completed" if user == "owner" else None),
            events=AsyncMock(return_value=[("completed", 19, {"type": "done", "content": "已完成"})]),
            cancel=AsyncMock(return_value=None),
        )
        with (
            patch.object(server.app.state, "chat_runs", store, create=True),
            patch.object(server, "verify_request"),
            patch.object(server, "authenticated_user", side_effect=lambda authorization: AuthenticatedUser(authorization)),
        ):
            client = TestClient(server.app)
            url = f"/chat/runs/{run_id}"
            self.assertEqual(client.get(url + "/events", headers={"Authorization": "other"}).status_code, 404)
            self.assertEqual(client.post(url + "/cancel", headers={"Authorization": "other"}).status_code, 404)
            response = client.get(url + "/events?after=18", headers={"Authorization": "owner"})
            self.assertEqual(response.status_code, 200)
            self.assertIn("id: 19", response.text)
            frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            self.assertEqual(frames[-1], {"type": "done", "content": "已完成", "event_id": "19"})
            store.events.assert_awaited_once_with("owner", run_id, 18)
