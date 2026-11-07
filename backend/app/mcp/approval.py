"""高风险 MCP 操作的短时、一次性人工确认记录。"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


SENSITIVE_ARGUMENTS = {"password", "secret", "token", "authorization", "api_key", "apikey"}


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def preview_arguments(arguments: dict) -> dict:
    """生成可向用户展示的参数预览，避免把常见凭据写入浏览器或日志。"""
    return {
        key: "[已隐藏]" if key.lower().replace("-", "_") in SENSITIVE_ARGUMENTS else value
        for key, value in arguments.items()
    }


class ApprovalStore:
    """SQLite 持久化审批；确认令牌只返回一次，库内仅保存哈希。"""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS mcp_approvals (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL, risk TEXT NOT NULL, token_hash TEXT NOT NULL,
                    question TEXT NOT NULL DEFAULT '', status TEXT NOT NULL, expires_at INTEGER NOT NULL, created_at INTEGER NOT NULL,
                    completed_at INTEGER, result TEXT
                )"""
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(mcp_approvals)")}
            if "question" not in columns:
                connection.execute("ALTER TABLE mcp_approvals ADD COLUMN question TEXT NOT NULL DEFAULT ''")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=5)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create(self, user_id: str, tool_name: str, arguments: dict, risk: str, question: str, ttl_seconds: int = 300) -> dict:
        approval_id, token, now = str(uuid.uuid4()), secrets.token_urlsafe(32), int(time.time())
        expires_at = now + ttl_seconds
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO mcp_approvals(id, user_id, tool_name, arguments_json, risk, token_hash, question, status, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                (approval_id, user_id, tool_name, json.dumps(arguments, ensure_ascii=False), risk, _token_hash(token), question, expires_at, now),
            )
        return {
            "approval_id": approval_id,
            "confirmation_token": token,
            "tool": tool_name,
            "risk": risk,
            "arguments": preview_arguments(arguments),
            "expires_at": expires_at,
        }

    def consume(self, approval_id: str, user_id: str, token: str) -> dict | None:
        """原子地消费待确认操作，避免双击或重放造成重复执行。"""
        now = int(time.time())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT user_id, tool_name, arguments_json, risk, token_hash, question, status, expires_at FROM mcp_approvals WHERE id = ?",
                (approval_id,),
            ).fetchone()
            if not row:
                return None
            stored_user, tool_name, arguments_json, risk, stored_hash, question, status, expires_at = row
            valid = (
                stored_user == user_id
                and status == "pending"
                and expires_at >= now
                and hmac.compare_digest(stored_hash, _token_hash(token))
            )
            if not valid:
                if status == "pending" and expires_at < now:
                    connection.execute("UPDATE mcp_approvals SET status = 'expired' WHERE id = ?", (approval_id,))
                return None
            changed = connection.execute(
                "UPDATE mcp_approvals SET status = 'executing' WHERE id = ? AND status = 'pending'",
                (approval_id,),
            ).rowcount
            if changed != 1:
                return None
        return {"tool": tool_name, "arguments": json.loads(arguments_json), "risk": risk, "question": question}

    def complete(self, approval_id: str, result: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE mcp_approvals SET status = 'completed', completed_at = ?, result = ? WHERE id = ? AND status = 'executing'",
                (int(time.time()), result, approval_id),
            )


class RedisApprovalStore:
    """Redis 持久化短时审批记录，Lua 原子消费，支持多实例。"""
    def __init__(self, redis_url: str):
        import redis
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)

    def _key(self, approval_id: str) -> str:
        return f"agent:approval:{approval_id}"

    def create(self, user_id: str, tool_name: str, arguments: dict, risk: str, question: str, ttl_seconds: int = 300) -> dict:
        approval_id, token, now = str(uuid.uuid4()), secrets.token_urlsafe(32), int(time.time())
        record = {"user_id": user_id, "tool": tool_name, "arguments": arguments, "risk": risk, "question": question, "token_hash": _token_hash(token), "status": "pending", "expires_at": now + ttl_seconds}
        self.client.set(self._key(approval_id), json.dumps(record, ensure_ascii=False), ex=ttl_seconds)
        return {"approval_id": approval_id, "confirmation_token": token, "tool": tool_name, "risk": risk, "arguments": preview_arguments(arguments), "expires_at": record["expires_at"]}

    def consume(self, approval_id: str, user_id: str, token: str) -> dict | None:
        script = """local raw=redis.call('GET',KEYS[1]); if not raw then return nil end; local v=cjson.decode(raw); if v.status~='pending' or v.user_id~=ARGV[1] or v.token_hash~=ARGV[2] then return nil end; v.status='executing'; redis.call('SET',KEYS[1],cjson.encode(v),'KEEPTTL'); return raw"""
        raw = self.client.eval(script, 1, self._key(approval_id), user_id, _token_hash(token))
        if not raw:
            return None
        record = json.loads(raw)
        return {key: record[key] for key in ("tool", "arguments", "risk", "question")}

    def complete(self, approval_id: str, result: str) -> None:
        script = """local raw=redis.call('GET',KEYS[1]); if not raw then return 0 end; local v=cjson.decode(raw); if v.status~='executing' then return 0 end; v.status='completed'; v.result=ARGV[1]; redis.call('SET',KEYS[1],cjson.encode(v),'KEEPTTL'); return 1"""
        self.client.eval(script, 1, self._key(approval_id), result)


def create_approval_store(database_path: Path):
    return RedisApprovalStore(__import__("os").environ["REDIS_URL"]) if __import__("os").environ.get("REDIS_URL") else ApprovalStore(database_path)
