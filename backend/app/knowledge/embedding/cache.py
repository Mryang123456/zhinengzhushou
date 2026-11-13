"""内容寻址的两级 Embedding 缓存。"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import unicodedata
import uuid
from collections import OrderedDict

from backend.app.knowledge.embedding.config import EmbeddingSettings


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def cache_key(settings: EmbeddingSettings, text: str, purpose: str, tenant_scope: str) -> str:
    identity = {
        "dimensions": settings.dimensions,
        "model": settings.model,
        "normalization": settings.normalization_version,
        "provider": settings.provider,
        "purpose": purpose,
        "revision": settings.model_revision,
        "tenant": tenant_scope,
        "text_sha256": hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"embedding:v3:{digest}"


class MemoryEmbeddingCache:
    def __init__(self, maximum: int, ttl_seconds: int):
        self.maximum, self.ttl_seconds = maximum, ttl_seconds
        self._values: OrderedDict[str, tuple[float, list[float]]] = OrderedDict()
        self._lock = threading.Lock()

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        now, found = time.monotonic(), {}
        with self._lock:
            for key in keys:
                value = self._values.get(key)
                if value is None:
                    continue
                expires_at, vector = value
                if expires_at <= now:
                    self._values.pop(key, None)
                    continue
                self._values.move_to_end(key)
                found[key] = list(vector)
        return found

    def set_many(self, values: dict[str, list[float]]) -> None:
        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            for key, vector in values.items():
                self._values[key] = (expires_at, list(vector))
                self._values.move_to_end(key)
            while len(self._values) > self.maximum:
                self._values.popitem(last=False)

    def delete_many(self, keys: set[str]) -> None:
        with self._lock:
            for key in keys:
                self._values.pop(key, None)


class TieredEmbeddingCache:
    def __init__(self, settings: EmbeddingSettings):
        self.local = MemoryEmbeddingCache(settings.cache_max_entries, settings.cache_ttl_seconds)
        self.ttl_seconds = settings.cache_ttl_seconds
        self.redis = None
        self._scope_keys: dict[str, set[str]] = {}
        self._scope_lock = threading.Lock()
        if settings.redis_url:
            import redis
            self.redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        found = self.local.get_many(keys)
        missing = [key for key in keys if key not in found]
        if self.redis is not None and missing:
            try:
                raw_values = self.redis.mget(missing)
                remote = {key: json.loads(raw) for key, raw in zip(missing, raw_values, strict=True) if raw is not None}
                self.local.set_many(remote)
                found.update(remote)
            except Exception:
                pass
        return found

    def delete_many(self, keys: set[str]) -> None:
        self.local.delete_many(keys)
        if self.redis is not None and keys:
            try:
                self.redis.delete(*keys)
            except Exception:
                pass

    def set_many(self, values: dict[str, list[float]]) -> None:
        self.local.set_many(values)
        if self.redis is not None and values:
            try:
                pipe = self.redis.pipeline(transaction=False)
                for key, vector in values.items():
                    pipe.setex(key, self.ttl_seconds, json.dumps(vector, separators=(",", ":")))
                pipe.execute()
            except Exception:
                pass

    def track_scope(self, tenant_scope: str, keys: set[str]) -> None:
        with self._scope_lock:
            if len(self._scope_keys) >= 10000 and tenant_scope not in self._scope_keys:
                self._scope_keys.pop(next(iter(self._scope_keys)))
            self._scope_keys.setdefault(tenant_scope, set()).update(keys)
        if self.redis is not None and keys:
            scope_hash = hashlib.sha256(tenant_scope.encode()).hexdigest()[:32]
            try:
                reference_key = f"embedding:refs:{scope_hash}"
                pipe = self.redis.pipeline(transaction=False)
                pipe.sadd(reference_key, *keys)
                pipe.expire(reference_key, self.ttl_seconds)
                pipe.execute()
            except Exception:
                pass

    def invalidate_scope(self, tenant_scope: str) -> int:
        with self._scope_lock:
            keys = self._scope_keys.pop(tenant_scope, set())
        if self.redis is not None:
            scope_hash = hashlib.sha256(tenant_scope.encode()).hexdigest()[:32]
            reference_key = f"embedding:refs:{scope_hash}"
            try:
                keys.update(self.redis.smembers(reference_key))
                pipe = self.redis.pipeline(transaction=True)
                if keys:
                    pipe.delete(*keys)
                pipe.delete(reference_key)
                pipe.execute()
            except Exception as error:
                raise ConnectionError("Embedding 缓存失效失败。") from error
        self.local.delete_many(keys)
        return len(keys)

    def acquire_lock(self, fingerprint: str, ttl_seconds: float) -> str | None:
        """Redis 可用时提供跨实例 SingleFlight；无 Redis 时由进程锁负责。"""
        if self.redis is None:
            return "local"
        token = uuid.uuid4().hex
        try:
            acquired = self.redis.set(
                f"embedding:lock:{fingerprint}", token, nx=True,
                ex=max(1, int(ttl_seconds) + 1),
            )
            return token if acquired else None
        except Exception as error:
            raise ConnectionError("Embedding 分布式锁不可用。") from error

    def release_lock(self, fingerprint: str, token: str | None) -> None:
        if self.redis is None or not token or token == "local":
            return
        script = "if redis.call('GET',KEYS[1])==ARGV[1] then return redis.call('DEL',KEYS[1]) else return 0 end"
        try:
            self.redis.eval(script, 1, f"embedding:lock:{fingerprint}", token)
        except Exception:
            pass

    def wait_many(self, keys: list[str], timeout_seconds: float) -> dict[str, list[float]]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            found = self.get_many(keys)
            if len(found) == len(keys):
                return found
            time.sleep(0.02)
        return self.get_many(keys)
