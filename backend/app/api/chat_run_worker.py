"""在独立于浏览器连接的后台任务中执行并恢复聊天工作流。"""

import asyncio
import logging
import os
from contextlib import aclosing, suppress

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import OperationalError

from backend.app.agent.service import build_agent
from backend.app.agent.turn_gate import thread_lock_key
from backend.app.agent.turns import checkpoint_config, stream_events
from backend.app.config import MCP_CONFIG_FILE, MEMORY_DB_FILE

logger = logging.getLogger(__name__)


async def _run_frames(store, conn, run_id, user_id, thread_id, message_id, payload, attempt):
    graph = await build_agent(
        MCP_CONFIG_FILE if payload["use_mcp"] else None,
        user_id,
        MEMORY_DB_FILE if payload["use_memory"] else None,
        os.getenv("MEMORY_DATABASE_URL") if payload["use_memory"] else None,
        AsyncPostgresSaver(conn),
        initial_question=payload["question"],
        recovering=attempt > 1,
    )
    tokens = []
    async with aclosing(stream_events(
        graph, payload["question"], thread_id, user_id, message_id=str(message_id),
        retry=payload["retry"], run_id=str(run_id),
    )) as frames:
        while True:
            next_frame = asyncio.create_task(anext(frames))
            try:
                while not next_frame.done():
                    await asyncio.wait({next_frame}, timeout=0.5)
                    if not next_frame.done() and await store.cancelled(run_id):
                        return {"type": "error", "message": "已停止生成。", "retryable": False}, ()
                frame = next_frame.result()
            finally:
                if not next_frame.done():
                    next_frame.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await next_frame
                elif not next_frame.cancelled():
                    next_frame.exception()
            if frame["type"] in {"done", "error"}:
                return frame, tokens if frame["type"] == "done" else ()
            if frame["type"] == "token":
                tokens.append(frame)
            else:
                await store.publish(run_id, attempt, frame)


async def _execute(store, conn, run_id, user_id, thread_id, claim):
    message_id, payload, attempts, cancelled = claim
    tokens = ()
    if cancelled:
        frame = {"type": "error", "message": "已停止生成。", "retryable": False}
    elif attempts > 3:
        frame = {"type": "error", "message": "任务多次中断，请重新发起。"}
    else:
        try:
            await store.append(conn, run_id, {"type": "progress", "stage": "preparing"})
            async with asyncio.timeout(float(os.getenv("AGENT_REQUEST_TIMEOUT_SECONDS", "90"))):
                frame, tokens = await _run_frames(store, conn, run_id, user_id, thread_id, message_id, payload, attempts)
        except TimeoutError:
            frame = {"type": "error", "message": "任务执行超时，请稍后重试。"}
        except OperationalError:
            # 数据库连接失效时保留任务，让下一个 worker 读取已提交的检查点。
            raise
        except Exception:
            logger.exception("后台聊天任务失败 run_id=%s", run_id)
            frame = {"type": "error", "message": "智能体执行失败，请稍后重试。"}
    await store.finish(conn, run_id, frame, tokens)


async def _process_next(store):
    async with store.pool.connection() as conn:
        after = 0
        while candidates := await store.candidates(conn, after):
            for after, run_id, user_id, thread_id in candidates:
                key = thread_lock_key(checkpoint_config(thread_id, user_id)["configurable"]["thread_id"])
                # 持锁连接写入 Checkpoint 和最终结果，进度事件通过执行次数隔离。
                locked = False
                try:
                    cursor = await conn.execute("SELECT pg_try_advisory_lock(%s::bigint)", (key,))
                    locked = (await cursor.fetchone())[0]
                    if not locked:
                        continue
                    claim = await store.claim(conn, run_id)
                    if claim:
                        await _execute(store, conn, run_id, user_id, thread_id, claim)
                        return True
                except asyncio.CancelledError:
                    # 取消可能发生在数据库已经加锁、但本地还没有读到结果时。
                    await conn.close()
                    raise
                except Exception:
                    if not locked:
                        await conn.close()
                    raise
                finally:
                    if locked and not conn.closed:
                        try:
                            await conn.execute("SELECT pg_advisory_unlock(%s::bigint)", (key,))
                        except BaseException:
                            await conn.close()
                            raise
    return False


async def run_chat_worker(store):
    while True:
        try:
            if await _process_next(store):
                continue
        except Exception:
            logger.exception("后台聊天队列暂时不可用")
        await asyncio.sleep(0.5)
