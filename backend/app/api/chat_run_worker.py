"""在独立于浏览器连接的后台任务中执行并恢复聊天工作流。"""

import asyncio
import logging
import os
import socket
from contextlib import aclosing, suppress
from uuid import uuid4

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import OperationalError

from backend.app.agent.service import build_agent
from backend.app.agent.turn_gate import thread_lock_key
from backend.app.agent.turns import checkpoint_config, stream_events
from backend.app.api.run_lease import LeaseLost, RunLeaseConfig, maintain_run_lease, stop_on_lease_loss
from backend.app.config import MCP_CONFIG_FILE, MEMORY_DB_FILE
from backend.app.observability import event, increment, trace_id
from backend.app.prompts import prompt_manifest_hash, prompt_scope
from backend.app.mcp.governance import mcp_enabled
from backend.app.security import account_enabled

logger = logging.getLogger(__name__)


async def _run_frames(store, conn, run_id, user_id, thread_id, message_id, payload, attempt, lease_token, lease_lost):
    manifest = payload.get("prompt_manifest")
    if not manifest:
        raise RuntimeError("聊天任务缺少Prompt Manifest，请重新发起，禁止使用当前版本静默续跑。")
    actual_hash = prompt_manifest_hash(manifest)
    if actual_hash != payload.get("prompt_manifest_hash"):
        raise RuntimeError("聊天任务的Prompt Manifest摘要不匹配，禁止执行。")
    event(
        "prompt_manifest_restored",
        run_id=str(run_id),
        prompt_manifest_hash=actual_hash,
        prompt_versions={item["prompt_id"]: item["version"] for item in manifest},
    )
    with prompt_scope(manifest):
        graph = await build_agent(
            MCP_CONFIG_FILE if payload["use_mcp"] and mcp_enabled() else None,
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
                        if lease_lost.is_set():
                            raise LeaseLost(f"任务 {run_id} 的心跳租约已失效。")
                        if not next_frame.done():
                            with suppress(TimeoutError):
                                async with asyncio.timeout(2):
                                    if await store.cancelled(run_id):
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
                    await store.publish(run_id, lease_token, frame)


async def _execute(store, conn, run_id, user_id, thread_id, claim, lease_config):
    message_id, payload, attempts, cancelled, lease_token = claim
    try:
        async with maintain_run_lease(store, run_id, lease_token, lease_config) as lease_lost, stop_on_lease_loss(lease_lost):
            tokens = ()
            if cancelled:
                frame = {"type": "error", "message": "已停止生成。", "retryable": False}
            elif not account_enabled(user_id):
                frame = {"type": "error", "message": "账号已停用。", "retryable": False}
            elif attempts > 3:
                increment("chat_recovery_exhausted")
                frame = {"type": "error", "message": "任务多次中断，请重新发起。"}
            else:
                try:
                    await store.publish(run_id, lease_token, {"type": "progress", "stage": "preparing"})
                    async with asyncio.timeout(float(os.getenv("AGENT_REQUEST_TIMEOUT_SECONDS", "90"))):
                        frame, tokens = await _run_frames(
                            store, conn, run_id, user_id, thread_id, message_id, payload,
                            attempts, lease_token, lease_lost,
                        )
                except TimeoutError:
                    frame = {"type": "error", "message": "任务执行超时，请稍后重试。"}
                except OperationalError:
                    # 数据库连接失效时保留任务，让其他 Worker 在租约过期后接管。
                    raise
                except LeaseLost:
                    raise
                except Exception:
                    logger.exception("后台聊天任务失败 run_id=%s", run_id)
                    frame = {"type": "error", "message": "智能体执行失败，请稍后重试。"}
            if lease_lost.is_set():
                raise LeaseLost(f"任务 {run_id} 已失去租约，禁止提交结果。")
            if frame["type"] == "error" and not cancelled:
                increment("chat_run_error")
            await store.finish(conn, run_id, lease_token, frame, tokens)
    except LeaseLost:
        event("chat_stale_worker_fenced", run_id=str(run_id), attempt=attempts)


async def _process_next(store, worker_id: str, lease_config: RunLeaseConfig):
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
                    claim = await store.claim(conn, run_id, worker_id, lease_config.duration_seconds)
                    if claim:
                        trace = trace_id.set(str(run_id))
                        try:
                            await _execute(store, conn, run_id, user_id, thread_id, claim, lease_config)
                        finally:
                            trace_id.reset(trace)
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
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex}"
    lease_config = RunLeaseConfig.from_env()
    while True:
        try:
            if await _process_next(store, worker_id, lease_config):
                continue
        except Exception:
            increment("chat_worker_error")
            logger.exception("后台聊天队列暂时不可用")
        await asyncio.sleep(0.5)
