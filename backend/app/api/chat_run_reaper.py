"""独立于执行 Worker 的过期任务回收循环。"""

import asyncio
import logging

from backend.app.api.run_lease import RunLeaseConfig
from backend.app.observability import event, increment

logger = logging.getLogger(__name__)


async def run_chat_reaper(store):
    config = RunLeaseConfig.from_env()
    while True:
        try:
            async with asyncio.timeout(config.reaper_seconds), store.pool.connection() as conn:
                expired = await store.reap_expired(conn)
            if expired:
                increment("chat_lease_reaped_runs", len(expired))
                event("chat_lease_reaped", count=len(expired), run_ids=[str(value) for value in expired])
        except Exception:
            increment("chat_reaper_error")
            logger.exception("过期聊天任务回收暂时失败")
        await asyncio.sleep(config.reaper_seconds)
