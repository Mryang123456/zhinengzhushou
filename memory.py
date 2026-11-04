"""本地 SQLite 长期记忆：显式保存、按用户隔离、可检索和可删除。"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


class MemoryStore(Protocol):
    def remember(self, user_id: str, content: str) -> int: ...
    def recent(self, user_id: str, limit: int = 12) -> list[tuple[int, str]]: ...
    def search(self, user_id: str, query: str, limit: int = 5) -> list[tuple[int, str]]: ...
    def forget(self, user_id: str, memory_id: int) -> bool: ...


class SQLiteMemoryStore:
    """不依赖外部服务的基础长期记忆存储。"""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    accessed_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, id DESC)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _contains_sensitive_value(content: str) -> bool:
        patterns = (r"\bsk-[A-Za-z0-9_-]+", r"\b(?:api[_-]?key|password|密码)\s*[:=]")
        return any(re.search(pattern, content, flags=re.IGNORECASE) for pattern in patterns)

    def remember(self, user_id: str, content: str) -> int:
        content = content.strip()
        if not content:
            raise ValueError("记忆内容不能为空。")
        if len(content) > 500:
            raise ValueError("单条记忆不能超过 500 个字符。")
        if self._contains_sensitive_value(content):
            raise ValueError("检测到疑似密钥、密码或 API Key，拒绝保存。")
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO memories(user_id, content, created_at, accessed_at) VALUES (?, ?, ?, ?)",
                (user_id, content, now, now),
            )
            return int(cursor.lastrowid)

    def recent(self, user_id: str, limit: int = 12) -> list[tuple[int, str]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id, content FROM memories WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [(int(row[0]), str(row[1])) for row in reversed(rows)]

    def search(self, user_id: str, query: str, limit: int = 5) -> list[tuple[int, str]]:
        # 基础版采用关键词和最近记录混合召回；后续可替换为向量检索。
        terms = [term.lower() for term in re.findall(r"[\w\u4e00-\u9fff]+", query) if len(term) > 1]
        candidates = self.recent(user_id, limit=50)
        ranked = sorted(
            candidates,
            key=lambda item: (
                sum(term in item[1].lower() for term in terms),
                item[0],
            ),
            reverse=True,
        )
        return ranked[:limit]

    def forget(self, user_id: str, memory_id: int) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM memories WHERE id = ? AND user_id = ?", (memory_id, user_id)
            )
            return cursor.rowcount == 1


class PostgresMemoryStore:
    """生产环境 PostgreSQL + pgvector 长期记忆存储。"""

    def __init__(self, database_url: str):
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("缺少 psycopg 依赖，请重新安装 requirements.txt。") from error
        self.database_url = database_url
        self._psycopg = psycopg
        self._initialize_schema()

    def _connect(self):
        return self._psycopg.connect(self.database_url)

    def _initialize_schema(self) -> None:
        schema_path = Path(__file__).with_name("db").joinpath("init.sql")
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(schema_path.read_text(encoding="utf-8"))

    def remember(self, user_id: str, content: str) -> int:
        content = content.strip()
        if not content:
            raise ValueError("记忆内容不能为空。")
        if len(content) > 500:
            raise ValueError("单条记忆不能超过 500 个字符。")
        if SQLiteMemoryStore._contains_sensitive_value(content):
            raise ValueError("检测到疑似密钥、密码或 API Key，拒绝保存。")
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO agent_memories(user_id, content) VALUES (%s, %s) RETURNING id",
                (user_id, content),
            )
            return int(cursor.fetchone()[0])

    def recent(self, user_id: str, limit: int = 12) -> list[tuple[int, str]]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, content FROM agent_memories
                WHERE user_id = %s AND deleted_at IS NULL
                ORDER BY id DESC LIMIT %s
                """,
                (user_id, limit),
            )
            rows = cursor.fetchall()
        return [(int(row[0]), str(row[1])) for row in reversed(rows)]

    def search(self, user_id: str, query: str, limit: int = 5) -> list[tuple[int, str]]:
        # pgvector embedding 列已预留；基础版先使用用户隔离后的关键词与近期记忆召回。
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, content FROM agent_memories
                WHERE user_id = %s AND deleted_at IS NULL
                ORDER BY
                    CASE WHEN content ILIKE %s THEN 1 ELSE 0 END DESC,
                    accessed_at DESC, id DESC
                LIMIT %s
                """,
                (user_id, f"%{query.strip()}%", limit),
            )
            rows = cursor.fetchall()
        return [(int(row[0]), str(row[1])) for row in rows]

    def forget(self, user_id: str, memory_id: int) -> bool:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE agent_memories SET deleted_at = NOW()
                WHERE id = %s AND user_id = %s AND deleted_at IS NULL
                """,
                (memory_id, user_id),
            )
            return cursor.rowcount == 1


def create_memory_store(
    database_url: str | None, sqlite_path: Path | None
) -> MemoryStore | None:
    """优先使用 PostgreSQL；本地开发可在未配置地址时回退到 SQLite。"""
    if database_url:
        return PostgresMemoryStore(database_url)
    if sqlite_path:
        return SQLiteMemoryStore(sqlite_path)
    return None
