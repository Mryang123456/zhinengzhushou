"""Embedding 缓存、分批、限流、熔断和重试的统一编排。"""

from __future__ import annotations

import hashlib
import math
import random
import threading
import time

from backend.app.knowledge.chunking import estimate_tokens
from backend.app.knowledge.embedding.cache import TieredEmbeddingCache, cache_key, normalize_text
from backend.app.knowledge.embedding.config import EmbeddingSettings
from backend.app.knowledge.embedding.providers import EmbeddingProvider
from backend.app.knowledge.embedding.resilience import (
    CircuitBreaker, EmbeddingCircuitOpen, EmbeddingRateLimited, SingleFlight, create_rate_limiter,
)
from backend.app.observability import event, increment, observe_duration


class EmbeddingGateway:
    def __init__(self, settings: EmbeddingSettings, provider: EmbeddingProvider):
        self.settings, self.provider = settings, provider
        self.cache = TieredEmbeddingCache(settings)
        self.rate_limiter = create_rate_limiter(
            settings.redis_url, settings.requests_per_minute, settings.tokens_per_minute,
            settings.tenant_requests_per_minute, settings.tenant_tokens_per_minute,
        )
        self.breaker = CircuitBreaker(settings.breaker_failures, settings.breaker_recovery_seconds)
        self.concurrency = threading.BoundedSemaphore(settings.max_concurrency)
        self.singleflight = SingleFlight()

    def embed(self, texts: list[str], *, purpose: str, tenant_scope: str) -> list[list[float]]:
        if purpose not in {"query", "document"} or not tenant_scope.strip():
            raise ValueError("Embedding purpose 或 tenant_scope 无效。")
        if len(texts) > 10000:
            raise ValueError("单次 Embedding 请求最多包含 10000 条文本。")
        normalized = [normalize_text(text) for text in texts]
        if not normalized or any(not text for text in normalized):
            raise ValueError("Embedding 文本不能为空。")
        token_counts = [estimate_tokens(text) for text in normalized]
        if any(count > self.settings.max_text_tokens for count in token_counts):
            raise ValueError("单条文本超过 Embedding Token 限制。")
        keys = [cache_key(self.settings, text, purpose, tenant_scope) for text in normalized]
        cached = self.cache.get_many(keys)
        invalid_keys = {key for key, vector in cached.items() if not self._valid_vector(vector)}
        if invalid_keys:
            self.cache.delete_many(invalid_keys)
            cached = {key: vector for key, vector in cached.items() if key not in invalid_keys}
        missing = {}
        for key, text, count in zip(keys, normalized, token_counts, strict=True):
            if key not in cached:
                missing.setdefault(key, (text, count))
        started = time.monotonic()
        if missing:
            self._fill_missing(missing, purpose, tenant_scope, cached)
        increment("embedding_items", len(texts))
        increment("embedding_cache_hits", len(texts) - len(missing))
        event("embedding_request", purpose=purpose, items=len(texts), cache_hits=len(texts) - len(missing))
        observe_duration("embedding", time.monotonic() - started)
        return [cached[key] for key in keys]

    def _fill_missing(
        self, missing: dict[str, tuple[str, int]], purpose: str,
        tenant_scope: str, output: dict[str, list[float]],
    ) -> None:
        batch = []
        batch_tokens = 0
        for key, (text, count) in missing.items():
            if batch and (len(batch) >= self.settings.max_batch_items or batch_tokens + count > self.settings.max_batch_tokens):
                self._run_batch(batch, purpose, tenant_scope, output)
                batch, batch_tokens = [], 0
            batch.append((key, text, count))
            batch_tokens += count
        if batch:
            self._run_batch(batch, purpose, tenant_scope, output)

    def _run_batch(
        self, batch: list[tuple[str, str, int]], purpose: str,
        tenant_scope: str, output: dict[str, list[float]],
    ) -> None:
        fingerprint = hashlib.sha256("|".join(key for key, _, _ in batch).encode()).hexdigest()
        wait_seconds = self.settings.timeout_seconds * self.settings.retry_attempts + 5
        if not self.singleflight.enter(fingerprint):
            self.singleflight.wait(fingerprint, wait_seconds)
            refreshed = self.cache.get_many([key for key, _, _ in batch])
            if len(refreshed) == len(batch):
                output.update(refreshed)
                return
            raise TimeoutError("等待相同 Embedding 请求完成超时。")
        distributed_token = None
        provider_attempted = False
        try:
            distributed_token = self.cache.acquire_lock(fingerprint, wait_seconds)
            if distributed_token is None:
                refreshed = self.cache.wait_many(
                    [key for key, _, _ in batch], wait_seconds,
                )
                if len(refreshed) == len(batch):
                    output.update(refreshed)
                    return
                raise TimeoutError("等待跨实例 Embedding 请求完成超时。")
            self.breaker.before_call()
            self.rate_limiter.acquire(tenant_scope, sum(count for _, _, count in batch))
            if not self.concurrency.acquire(timeout=self.settings.timeout_seconds):
                raise TimeoutError("等待 Embedding 并发额度超时。")
            try:
                provider_attempted = True
                vectors = self._call_with_retry([text for _, text, _ in batch], purpose, tenant_scope)
            finally:
                self.concurrency.release()
            if len(vectors) != len(batch) or any(not self._valid_vector(vector) for vector in vectors):
                raise RuntimeError("Embedding 返回数量或维度不符合约定。")
            values = {key: [float(value) for value in vector] for (key, _, _), vector in zip(batch, vectors, strict=True)}
            self.cache.set_many(values)
            self.cache.track_scope(tenant_scope, set(values))
            output.update(values)
            increment("embedding_provider_batches")
            increment("embedding_provider_tokens", sum(count for _, _, count in batch))
            self.breaker.success()
        except (EmbeddingCircuitOpen, EmbeddingRateLimited):
            self.breaker.cancel_probe()
            event("embedding_rejected", purpose=purpose)
            raise
        except ValueError:
            self.breaker.cancel_probe()
            event("embedding_invalid", purpose=purpose)
            raise
        except Exception:
            if provider_attempted:
                self.breaker.failure()
            else:
                self.breaker.cancel_probe()
            event("embedding_failure", purpose=purpose)
            raise
        finally:
            self.cache.release_lock(fingerprint, distributed_token)
            self.singleflight.leave(fingerprint)

    def _call_with_retry(self, texts: list[str], purpose: str, tenant_scope: str) -> list[list[float]]:
        last_error = None
        for attempt in range(self.settings.retry_attempts):
            try:
                return self.provider.embed(texts, purpose, tenant_scope)
            except (TimeoutError, ConnectionError, OSError, RuntimeError) as error:
                last_error = error
                if attempt + 1 < self.settings.retry_attempts:
                    time.sleep(min(0.2 * 2 ** attempt + random.random() * 0.1, 2.0))
        raise last_error or RuntimeError("Embedding 调用失败。")

    def invalidate_scope(self, tenant_scope: str) -> int:
        if not tenant_scope.strip():
            raise ValueError("tenant_scope 不能为空。")
        invalidated = self.cache.invalidate_scope(tenant_scope)
        remote_invalidator = getattr(self.provider, "invalidate_scope", None)
        if remote_invalidator is not None:
            invalidated += remote_invalidator(tenant_scope)
        return invalidated

    def _valid_vector(self, vector: object) -> bool:
        return (
            isinstance(vector, list)
            and len(vector) == self.settings.dimensions
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in vector)
        )
