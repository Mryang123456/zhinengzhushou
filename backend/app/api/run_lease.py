"""聊天任务的分布式租约配置、心跳监督和 fencing 异常。"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass

from backend.app.observability import event, increment


logger = logging.getLogger(__name__)


class LeaseLost(RuntimeError):
    """当前 Worker 的租约已过期、被回收或被其他 Worker 接管。"""


@asynccontextmanager
async def stop_on_lease_loss(lost):
    """失租时取消当前执行，包括正在等待的数据库或模型请求。"""
    owner = asyncio.current_task()

    async def watch():
        await lost.wait()
        owner.cancel()

    watcher = asyncio.create_task(watch())
    try:
        if lost.is_set():
            raise LeaseLost("任务租约已失效。")
        yield
    except asyncio.CancelledError:
        if lost.is_set():
            raise LeaseLost("任务租约已失效，已中断执行。") from None
        raise
    finally:
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


@dataclass(frozen=True)
class RunLeaseConfig:
    duration_seconds: float
    heartbeat_seconds: float
    reaper_seconds: float

    @classmethod
    def from_env(cls) -> "RunLeaseConfig":
        duration = _bounded_float("AGENT_RUN_LEASE_SECONDS", 30, 5, 300)
        heartbeat = _bounded_float("AGENT_RUN_HEARTBEAT_SECONDS", duration / 3, 1, 60)
        heartbeat = min(heartbeat, max(1, duration / 2))
        reaper = _bounded_float("AGENT_RUN_REAPER_SECONDS", 5, 1, 60)
        return cls(duration, heartbeat, reaper)


@asynccontextmanager
async def maintain_run_lease(store, run_id, lease_token, config: RunLeaseConfig):
    """独立连接续租；数据库短暂失败时在本地截止时间前重试。"""
    lost = asyncio.Event()
    stopped = asyncio.Event()

    async def heartbeat_loop():
        local_deadline = time.monotonic() + config.duration_seconds
        while not stopped.is_set():
            try:
                await asyncio.wait_for(stopped.wait(), timeout=config.heartbeat_seconds)
                return
            except TimeoutError:
                pass
            try:
                remaining = local_deadline - time.monotonic()
                if remaining <= 0:
                    renewed = False
                else:
                    async with asyncio.timeout(remaining):
                        renewed = await store.heartbeat(run_id, lease_token, config.duration_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                increment("chat_lease_heartbeat_error")
                logger.warning("聊天任务心跳失败 run_id=%s error=%s", run_id, type(error).__name__)
                if time.monotonic() < local_deadline:
                    continue
                renewed = False
            if renewed:
                local_deadline = time.monotonic() + config.duration_seconds
                increment("chat_lease_heartbeat")
                continue
            lost.set()
            event("chat_lease_lost", run_id=str(run_id))
            return

    task = asyncio.create_task(heartbeat_loop(), name=f"chat-lease-{run_id}")
    try:
        yield lost
    finally:
        stopped.set()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
