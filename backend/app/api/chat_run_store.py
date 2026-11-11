"""PostgreSQL 聊天任务和可重放的事件记录。"""

from pathlib import Path
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb

from backend.app.agent.turn_gate import thread_lock_key
from backend.app.config import BACKEND_ROOT

TERMINAL = {"completed", "failed", "cancelled"}


class SubmissionConflict(ValueError):
    pass


class ChatRunStore:
    def __init__(self, pool):
        self.pool = pool

    async def setup(self):
        schema = Path(BACKEND_ROOT, "db", "chat_runs.sql").read_text(encoding="utf-8")
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(764814924191)")
            await conn.execute(schema, prepare=False)

    async def submit(self, user_id, thread_id, message_id, request_id, payload) -> UUID:
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(%s::bigint)", (thread_lock_key(f"submit:{user_id}:{thread_id}"),))
            cursor = await conn.execute(
                """
                INSERT INTO chat_runs (id, user_id, thread_id, message_id, request_id, payload)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, thread_id, request_id) DO NOTHING RETURNING id
                """,
                (uuid4(), user_id, thread_id, message_id, request_id, Jsonb(payload)),
            )
            inserted = await cursor.fetchone()
            if inserted:
                return inserted[0]
            cursor = await conn.execute(
                "SELECT id, message_id, payload FROM chat_runs WHERE user_id = %s AND thread_id = %s AND request_id = %s",
                (user_id, thread_id, request_id),
            )
            existing = await cursor.fetchone()
            if existing[1:] != (message_id, payload):
                raise SubmissionConflict("该请求 ID 已用于其他消息或参数。")
            return existing[0]

    async def status(self, user_id, run_id):
        async with self.pool.connection() as conn:
            cursor = await conn.execute("SELECT status FROM chat_runs WHERE id = %s AND user_id = %s", (run_id, user_id))
            row = await cursor.fetchone()
            return row[0] if row else None

    async def approval_token(self, user_id: str, run_id: UUID, approval_id: str) -> str | None:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT event.payload ->> 'confirmation_token'
                FROM chat_run_events event JOIN chat_runs run ON run.id = event.run_id
                WHERE run.id = %s AND run.user_id = %s
                    AND event.payload ->> 'type' = 'approval_required'
                    AND event.payload ->> 'approval_id' = %s
                ORDER BY event.id DESC LIMIT 1
                """,
                (run_id, user_id, approval_id),
            )
            row = await cursor.fetchone()
            return row[0] if row else None

    async def cancel(self, user_id, run_id):
        async with self.pool.connection() as conn, conn.transaction():
            cursor = await conn.execute(
                "SELECT status FROM chat_runs WHERE id = %s AND user_id = %s FOR UPDATE",
                (run_id, user_id),
            )
            row = await cursor.fetchone()
            if row and row[0] not in TERMINAL:
                await conn.execute("UPDATE chat_runs SET cancel_requested = TRUE WHERE id = %s", (run_id,))
                if row[0] == "queued":
                    await self.finish(conn, run_id, {"type": "error"})
            return row[0] if row else None

    async def events(self, user_id, run_id, after):
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT r.status, event.id, event.payload
                FROM chat_runs r LEFT JOIN LATERAL (
                    SELECT id, payload FROM chat_run_events WHERE run_id = r.id AND id > %s ORDER BY id LIMIT 100
                ) event ON TRUE WHERE r.id = %s AND r.user_id = %s ORDER BY event.id
                """,
                (after, run_id, user_id),
            )
            return await cursor.fetchall()

    @staticmethod
    async def candidates(conn, after):
        cursor = await conn.execute(
            """
            SELECT r.position, r.id, r.user_id, r.thread_id
            FROM chat_runs r WHERE r.status IN ('queued', 'running') AND r.position > %s
            AND NOT EXISTS (
                SELECT 1 FROM chat_runs older
                WHERE older.user_id = r.user_id AND older.thread_id = r.thread_id AND older.position < r.position
                AND older.status IN ('queued', 'running')
            ) ORDER BY r.position LIMIT 32
            """,
            (after,),
        )
        return await cursor.fetchall()

    @staticmethod
    async def claim(conn, run_id):
        cursor = await conn.execute(
            """
            UPDATE chat_runs SET status = 'running', attempts = attempts + 1
            WHERE id = %s AND status IN ('queued', 'running')
            RETURNING message_id, payload, attempts, cancel_requested
            """,
            (run_id,),
        )
        return await cursor.fetchone()

    async def cancelled(self, run_id):
        async with self.pool.connection() as conn:
            cursor = await conn.execute("SELECT cancel_requested FROM chat_runs WHERE id = %s", (run_id,))
            return (await cursor.fetchone())[0]

    async def publish(self, run_id, attempt, frame):
        """使用独立连接记录进度，避免与 LangGraph 的数据库流水线共用连接。"""
        async with self.pool.connection() as conn, conn.transaction():
            cursor = await conn.execute(
                "SELECT id FROM chat_runs WHERE id = %s AND attempts = %s AND status = 'running' FOR UPDATE",
                (run_id, attempt),
            )
            if await cursor.fetchone():
                await self.append(conn, run_id, frame)

    @staticmethod
    async def append(conn, run_id, frame):
        await conn.execute("INSERT INTO chat_run_events(run_id, payload) VALUES (%s, %s)", (run_id, Jsonb(frame)))

    async def finish(self, conn, run_id, frame, tokens=()):
        async with conn.transaction():
            cursor = await conn.execute("SELECT cancel_requested FROM chat_runs WHERE id = %s FOR UPDATE", (run_id,))
            if (await cursor.fetchone())[0]:
                frame = {"type": "error", "message": "已停止生成。", "retryable": False}
                status, tokens = "cancelled", ()
            else:
                status = "completed" if frame["type"] == "done" else "failed"
            for token in tokens:
                await self.append(conn, run_id, token)
            await self.append(conn, run_id, frame)
            await conn.execute("UPDATE chat_runs SET status = %s, finished_at = NOW() WHERE id = %s", (status, run_id))
