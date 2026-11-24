"""有时限的服务依赖检查。"""

import asyncio


async def _postgres(pool):
    async with pool.connection() as connection:
        await connection.execute("SELECT 1")
    return True


async def _bounded(probe):
    try:
        async with asyncio.timeout(2):
            return bool(await probe())
    except Exception:
        return False


async def dependency_status(state, *, postgres_required: bool, redis_required: bool):
    checks = {}
    if postgres_required:
        pool = getattr(state, "checkpoint_pool", None)
        workers = getattr(state, "chat_workers", ())
        checks["postgresql"] = bool(pool) and await _bounded(lambda: _postgres(pool))
        checks["chat_workers"] = bool(workers) and all(not task.done() for task in workers)
    if redis_required:
        redis = getattr(state, "redis_probe", None)
        checks["redis"] = bool(redis) and await _bounded(redis.ping)
    return checks


async def queue_metrics(pool):
    if pool is None:
        return ""
    try:
        async with asyncio.timeout(2), pool.connection() as connection:
            cursor = await connection.execute("""
                SELECT
                    count(*) FILTER (WHERE status = 'queued'),
                    coalesce(extract(epoch FROM now() - min(created_at) FILTER (WHERE status = 'queued')), 0),
                    count(*) FILTER (WHERE status = 'running' AND lease_expires_at > clock_timestamp()),
                    count(*) FILTER (WHERE status = 'running'
                        AND (lease_expires_at IS NULL OR lease_expires_at <= clock_timestamp()))
                FROM chat_runs WHERE status IN ('queued', 'running')
            """)
            count, oldest, active_leases, expired_leases = await cursor.fetchone()
    except Exception:
        return ""
    return (
        f"agent_chat_queue_depth {count}\n"
        f"agent_chat_queue_oldest_seconds {max(0, oldest)}\n"
        f"agent_chat_active_leases {active_leases}\n"
        f"agent_chat_expired_leases {expired_leases}\n"
    )
