"""Embedding 调用的限流、并发保护、熔断和请求合并。"""

from __future__ import annotations

import threading
import time
import hashlib
from dataclasses import dataclass


class EmbeddingRateLimited(RuntimeError):
    pass


class EmbeddingCircuitOpen(RuntimeError):
    pass


class TokenBucket:
    def __init__(self, capacity: int, refill_per_second: float):
        self.capacity = float(capacity)
        self.refill_per_second = refill_per_second
        self.tokens = float(capacity)
        self.updated_at = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, amount: int) -> bool:
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated_at) * self.refill_per_second)
            self.updated_at = now
            if amount > self.tokens:
                return False
            self.tokens -= amount
            return True

    def refund(self, amount: int) -> None:
        with self.lock:
            self.tokens = min(self.capacity, self.tokens + amount)


class RateLimiter:
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests = TokenBucket(requests_per_minute, requests_per_minute / 60)
        self.tokens = TokenBucket(tokens_per_minute, tokens_per_minute / 60)

    def acquire(self, token_count: int) -> None:
        if not self.requests.consume(1):
            raise EmbeddingRateLimited("Embedding 服务额度暂时不足。")
        if not self.tokens.consume(token_count):
            self.requests.refund(1)
            raise EmbeddingRateLimited("Embedding 服务额度暂时不足。")

    def refund(self, token_count: int) -> None:
        self.requests.refund(1)
        self.tokens.refund(token_count)


class ScopedRateLimiter:
    """同时限制全局额度和租户额度，防止单租户抢占全部容量。"""
    def __init__(
        self, requests_per_minute: int, tokens_per_minute: int,
        tenant_requests_per_minute: int, tenant_tokens_per_minute: int,
    ):
        self.tenant_requests_per_minute = tenant_requests_per_minute
        self.tenant_tokens_per_minute = tenant_tokens_per_minute
        self.global_limiter = RateLimiter(requests_per_minute, tokens_per_minute)
        self._tenants: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

    def acquire(self, identity: str, token_count: int) -> None:
        with self._lock:
            limiter = self._tenants.get(identity)
            if limiter is None:
                if len(self._tenants) >= 10000:
                    self._tenants.pop(next(iter(self._tenants)))
                limiter = self._tenants.setdefault(
                    identity, RateLimiter(self.tenant_requests_per_minute, self.tenant_tokens_per_minute),
                )
        self.global_limiter.acquire(token_count)
        try:
            limiter.acquire(token_count)
        except Exception:
            self.global_limiter.refund(token_count)
            raise


class RedisScopedRateLimiter:
    """Redis Lua 原子限制跨实例的全局及租户 RPM/TPM。"""
    SCRIPT = """
    local values={}
    for i,key in ipairs(KEYS) do values[i]=tonumber(redis.call('GET',key) or '0') end
    if values[1]+1>tonumber(ARGV[1]) or values[2]+tonumber(ARGV[3])>tonumber(ARGV[2])
       or values[3]+1>tonumber(ARGV[5]) or values[4]+tonumber(ARGV[3])>tonumber(ARGV[6]) then return 0 end
    for i,key in ipairs(KEYS) do
      local amount=(i==1 or i==3) and 1 or tonumber(ARGV[3])
      redis.call('INCRBY',key,amount); redis.call('EXPIRE',key,tonumber(ARGV[4]))
    end
    return 1
    """

    def __init__(
        self, redis_url: str, requests_per_minute: int, tokens_per_minute: int,
        tenant_requests_per_minute: int, tenant_tokens_per_minute: int,
    ):
        import redis
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.requests_per_minute, self.tokens_per_minute = requests_per_minute, tokens_per_minute
        self.tenant_requests_per_minute = tenant_requests_per_minute
        self.tenant_tokens_per_minute = tenant_tokens_per_minute

    def acquire(self, identity: str, token_count: int) -> None:
        bucket = int(time.time() // 60)
        tenant = hashlib.sha256(identity.encode()).hexdigest()[:24]
        prefix = f"embedding:rate:{bucket}"
        keys = [f"{prefix}:global:r", f"{prefix}:global:t", f"{prefix}:{tenant}:r", f"{prefix}:{tenant}:t"]
        allowed = self.client.eval(
            self.SCRIPT, len(keys), *keys,
            self.requests_per_minute, self.tokens_per_minute, token_count, 61,
            self.tenant_requests_per_minute, self.tenant_tokens_per_minute,
        )
        if not allowed:
            raise EmbeddingRateLimited("Embedding 服务额度暂时不足。")


def create_rate_limiter(
    redis_url: str, requests_per_minute: int, tokens_per_minute: int,
    tenant_requests_per_minute: int, tenant_tokens_per_minute: int,
):
    if redis_url:
        return RedisScopedRateLimiter(
            redis_url, requests_per_minute, tokens_per_minute,
            tenant_requests_per_minute, tenant_tokens_per_minute,
        )
    return ScopedRateLimiter(
        requests_per_minute, tokens_per_minute,
        tenant_requests_per_minute, tenant_tokens_per_minute,
    )


@dataclass
class CircuitBreaker:
    failure_threshold: int
    recovery_seconds: float
    failures: int = 0
    opened_at: float | None = None
    probe_in_flight: bool = False

    def __post_init__(self):
        self.lock = threading.Lock()

    def before_call(self) -> None:
        with self.lock:
            if self.opened_at is None:
                return
            if time.monotonic() - self.opened_at < self.recovery_seconds or self.probe_in_flight:
                raise EmbeddingCircuitOpen("Embedding 上游熔断中。")
            self.probe_in_flight = True

    def success(self) -> None:
        with self.lock:
            self.failures, self.opened_at, self.probe_in_flight = 0, None, False

    def failure(self) -> None:
        with self.lock:
            self.failures += 1
            self.probe_in_flight = False
            if self.failures >= self.failure_threshold:
                self.opened_at = time.monotonic()

    def cancel_probe(self) -> None:
        """限流或输入错误不计入上游故障，并释放半开探针。"""
        with self.lock:
            self.probe_in_flight = False


class SingleFlight:
    """同进程相同批次只允许一个调用者访问上游。"""
    def __init__(self):
        self._condition = threading.Condition()
        self._active: set[str] = set()

    def enter(self, key: str) -> bool:
        with self._condition:
            if key in self._active:
                return False
            self._active.add(key)
            return True

    def wait(self, key: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while key in self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._condition.wait(remaining)

    def leave(self, key: str) -> None:
        with self._condition:
            self._active.discard(key)
            self._condition.notify_all()
