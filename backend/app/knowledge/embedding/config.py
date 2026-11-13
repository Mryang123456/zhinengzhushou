"""Embedding 配置及边界校验。"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


def _integer(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间。")
    return value


def _float(name: str, default: float, minimum: float, maximum: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间。")
    return value


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    model_revision: str
    dimensions: int
    service_url: str
    service_api_key: str
    timeout_seconds: float
    max_batch_items: int
    max_batch_tokens: int
    max_text_tokens: int
    max_concurrency: int
    requests_per_minute: int
    tokens_per_minute: int
    tenant_requests_per_minute: int
    tenant_tokens_per_minute: int
    retry_attempts: int
    cache_ttl_seconds: int
    cache_max_entries: int
    normalization_version: str
    breaker_failures: int
    breaker_recovery_seconds: float
    redis_url: str

    def for_purpose(self, purpose: str) -> "EmbeddingSettings":
        if purpose != "query":
            return self
        return replace(
            self,
            timeout_seconds=_float("KNOWLEDGE_EMBEDDING_QUERY_TIMEOUT_SECONDS", 5.0, 0.1, 60.0),
            retry_attempts=_integer("KNOWLEDGE_EMBEDDING_QUERY_RETRY_ATTEMPTS", 1, 1, 3),
        )

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        provider = os.getenv("KNOWLEDGE_EMBEDDING_PROVIDER", "").strip().lower() or "disabled"
        if provider not in {"disabled", "openai"}:
            raise ValueError("KNOWLEDGE_EMBEDDING_PROVIDER 仅支持 disabled 或 openai。")
        settings = cls(
            provider=provider,
            model=os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
            model_revision=os.getenv("KNOWLEDGE_EMBEDDING_MODEL_REVISION", "stable").strip(),
            dimensions=_integer("KNOWLEDGE_EMBEDDING_DIMENSIONS", 1536, 1, 65536),
            service_url=os.getenv("KNOWLEDGE_EMBEDDING_SERVICE_URL", "").strip().rstrip("/"),
            service_api_key=os.getenv("EMBEDDING_SERVICE_API_KEY", ""),
            timeout_seconds=_float("KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS", 8.0, 0.1, 120.0),
            max_batch_items=_integer("KNOWLEDGE_EMBEDDING_BATCH_ITEMS", 64, 1, 2048),
            max_batch_tokens=_integer("KNOWLEDGE_EMBEDDING_BATCH_TOKENS", 20000, 1, 2_000_000),
            max_text_tokens=_integer("KNOWLEDGE_EMBEDDING_MAX_TEXT_TOKENS", 8000, 1, 1_000_000),
            max_concurrency=_integer("KNOWLEDGE_EMBEDDING_MAX_CONCURRENCY", 4, 1, 128),
            requests_per_minute=_integer("KNOWLEDGE_EMBEDDING_RPM", 120, 1, 1_000_000),
            tokens_per_minute=_integer("KNOWLEDGE_EMBEDDING_TPM", 500000, 1, 100_000_000),
            tenant_requests_per_minute=_integer("KNOWLEDGE_EMBEDDING_TENANT_RPM", 60, 1, 1_000_000),
            tenant_tokens_per_minute=_integer("KNOWLEDGE_EMBEDDING_TENANT_TPM", 100000, 1, 100_000_000),
            retry_attempts=_integer("KNOWLEDGE_EMBEDDING_RETRY_ATTEMPTS", 3, 1, 8),
            cache_ttl_seconds=_integer("KNOWLEDGE_EMBEDDING_CACHE_TTL_SECONDS", 2592000, 1, 31_536_000),
            cache_max_entries=_integer("KNOWLEDGE_EMBEDDING_CACHE_MAX_ENTRIES", 10000, 1, 1_000_000),
            normalization_version=os.getenv("KNOWLEDGE_EMBEDDING_NORMALIZATION_VERSION", "nfc-v1").strip(),
            breaker_failures=_integer("KNOWLEDGE_EMBEDDING_BREAKER_FAILURES", 5, 1, 100),
            breaker_recovery_seconds=_float("KNOWLEDGE_EMBEDDING_BREAKER_RECOVERY_SECONDS", 30.0, 1.0, 3600.0),
            redis_url=os.getenv("REDIS_URL", "").strip(),
        )
        if settings.service_url and not settings.service_api_key:
            raise ValueError("配置独立 Embedding 服务时必须设置 EMBEDDING_SERVICE_API_KEY。")
        if settings.dimensions != 1536:
            raise ValueError("当前 knowledge_chunks.embedding 数据库列要求 1536 维向量。")
        if (
            settings.tenant_requests_per_minute > settings.requests_per_minute
            or settings.tenant_tokens_per_minute > settings.tokens_per_minute
        ):
            raise ValueError("租户级 Embedding 限额不能高于全局限额。")
        return settings
