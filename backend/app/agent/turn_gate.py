"""按 Checkpoint 线程串行执行聊天请求。"""

import asyncio
import hashlib
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field


class TurnQueueTimeout(TimeoutError):
    """等待上一轮完成超时。"""


@dataclass
class _ThreadQueue:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


def thread_lock_key(thread_key: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(thread_key.encode("utf-8"), digest_size=8, person=b"zhixing-turn").digest(),
        "big", signed=True,
    )


@asynccontextmanager
async def _postgres_turn_lock(pool, thread_key: str, wait_seconds: float):
    key = thread_lock_key(thread_key)
    async with pool.connection(timeout=wait_seconds) as connection:
        acquired = False
        try:
            await connection.execute("SELECT pg_advisory_lock(%s::bigint)", (key,))
            acquired = True
            yield connection
        finally:
            if not acquired:
                # 获取锁时被取消，连接不能带着状态未知的锁回到池中。
                await connection.close()
            else:
                try:
                    async with asyncio.timeout(3):
                        await connection.execute("SELECT pg_advisory_unlock(%s::bigint)", (key,))
                except BaseException:
                    await connection.close()
                    raise


class ThreadTurnGate:
    def __init__(self):
        self._queues: dict[str, _ThreadQueue] = {}

    @asynccontextmanager
    async def hold(self, thread_key: str, wait_seconds: float, pool=None):
        queue = self._queues.get(thread_key)
        if queue is None:
            queue = self._queues[thread_key] = _ThreadQueue()
        queue.users += 1
        try:
            async with AsyncExitStack() as resources:
                try:
                    async with asyncio.timeout(wait_seconds):
                        await resources.enter_async_context(queue.lock)
                        connection = (
                            await resources.enter_async_context(_postgres_turn_lock(pool, thread_key, wait_seconds))
                            if pool is not None else None
                        )
                except TimeoutError as error:
                    raise TurnQueueTimeout from error
                yield connection
        finally:
            queue.users -= 1
            if queue.users == 0:
                del self._queues[thread_key]
