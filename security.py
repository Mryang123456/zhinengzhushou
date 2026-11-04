"""服务端请求鉴权与内存限流。生产环境应替换为网关或 Redis 限流。"""

from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque
from threading import Lock


class SlidingWindowRateLimiter:
    """按客户端标识限制固定时间窗口内的请求数。"""

    def __init__(self, limit: int = 20, window_seconds: int = 60):
        if limit < 1 or window_seconds < 1:
            raise ValueError("限流参数必须为正数。")
        self.limit = limit
        self.window_seconds = window_seconds
        self._records: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, identity: str, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        with self._lock:
            records = self._records[identity]
            threshold = current - self.window_seconds
            while records and records[0] <= threshold:
                records.popleft()
            if len(records) >= self.limit:
                return False
            records.append(current)
            return True


def valid_api_key(provided_key: str | None, expected_key: str | None) -> bool:
    """未配置服务密钥时仅允许本地开发；配置后使用恒定时间比较。"""
    if not expected_key:
        return True
    return bool(provided_key) and hmac.compare_digest(
        provided_key.encode("utf-8"), expected_key.encode("utf-8")
    )
