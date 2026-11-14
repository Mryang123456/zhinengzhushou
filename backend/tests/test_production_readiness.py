"""生产探针、账号停用与 MCP 启用边界。"""

import os
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend.app.api import server
from backend.app.mcp.governance import ToolPolicyError, load_mcp_connections, mcp_enabled
from backend.app.security import AuthenticationError, AuthenticatedUser, authenticate_bearer, issue_access_token


class ReadinessTests(unittest.TestCase):
    def test_configured_dependencies_change_http_readiness(self):
        connection = SimpleNamespace(execute=AsyncMock())

        @asynccontextmanager
        async def acquire():
            yield connection

        pool = SimpleNamespace(connection=acquire)
        redis = SimpleNamespace(ping=AsyncMock(return_value=True))
        worker = SimpleNamespace(done=lambda: False)
        with (
            patch.dict(os.environ, {"CHECKPOINT_DATABASE_URL": "test", "REDIS_URL": "test"}),
            patch.object(server.app.state, "checkpoint_pool", pool, create=True),
            patch.object(server.app.state, "redis_probe", redis, create=True),
            patch.object(server.app.state, "chat_workers", [worker], create=True),
        ):
            client = TestClient(server.app)
            self.assertEqual(client.get("/readyz").json(), {
                "status": "ready", "checks": {"postgresql": True, "chat_workers": True, "redis": True},
            })
            redis.ping.side_effect = OSError("private host")
            worker.done = lambda: True
            connection.execute.side_effect = OSError("private database")
            response = client.get("/readyz")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["checks"], {"postgresql": False, "chat_workers": False, "redis": False})
            self.assertNotIn("private", response.text)
            self.assertEqual(client.get("/healthz").status_code, 200)

    def test_token_version_rotation_and_account_disable_invalidate_tokens(self):
        with patch.dict(os.environ, {"AUTH_REQUIRED": "true", "JWT_SECRET": "x" * 32,
                                     "AUTH_DISABLED_USERS": "", "JWT_TOKEN_VERSION": "1"}):
            token = "Bearer " + issue_access_token(AuthenticatedUser("账号甲"))
            self.assertEqual(authenticate_bearer(token).user_id, "账号甲")
            os.environ["AUTH_DISABLED_USERS"] = "账号甲"
            with self.assertRaises(AuthenticationError):
                authenticate_bearer(token)
            os.environ["AUTH_DISABLED_USERS"] = ""
            os.environ["JWT_TOKEN_VERSION"] = "2"
            with self.assertRaises(AuthenticationError):
                authenticate_bearer(token)

    def test_production_mcp_is_opt_in_and_rejects_local_registry(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True):
            self.assertFalse(mcp_enabled())
            os.environ["AGENT_MCP_ENABLED"] = "true"
            self.assertTrue(mcp_enabled())
            with self.assertRaises(ToolPolicyError):
                load_mcp_connections(server.MCP_CONFIG_FILE)
