"""知识库 Embedding 门面，支持进程内治理和独立服务两种部署模式。"""

from __future__ import annotations

from functools import lru_cache

from backend.app.knowledge.embedding.config import EmbeddingSettings
from backend.app.knowledge.embedding.gateway import EmbeddingGateway
from backend.app.knowledge.embedding.providers import OpenAIEmbeddingProvider, RemoteEmbeddingProvider


class EmbeddingUnavailable(RuntimeError):
    """Embedding 未启用或暂时不可用。"""


@lru_cache(maxsize=4)
def _gateway(settings: EmbeddingSettings) -> EmbeddingGateway:
    provider = RemoteEmbeddingProvider(settings) if settings.service_url else OpenAIEmbeddingProvider(settings)
    return EmbeddingGateway(settings, provider)


def reset_embedding_gateway() -> None:
    """清理进程内实例，供配置热加载和测试使用。"""
    _gateway.cache_clear()


def invalidate_embedding_cache(tenant_scope: str) -> int:
    """主动清除租户缓存，供文档隐私删除或租户注销流程调用。"""
    base = EmbeddingSettings.from_env()
    if base.provider == "disabled":
        return 0
    document_gateway = _gateway(base.for_purpose("document"))
    query_gateway = _gateway(base.for_purpose("query"))
    invalidated = document_gateway.invalidate_scope(tenant_scope)
    if query_gateway is not document_gateway:
        invalidated += query_gateway.cache.invalidate_scope(tenant_scope)
    return invalidated


def embed_texts(
    texts: list[str], *, for_search: bool = False, tenant_scope: str = "system",
) -> list[list[float]]:
    """批量生成向量；查询失败时抛出可降级异常。"""
    purpose = "query" if for_search else "document"
    settings = EmbeddingSettings.from_env().for_purpose(purpose)
    if settings.provider == "disabled":
        raise EmbeddingUnavailable("未启用 Embedding；将使用关键词检索。")
    try:
        return _gateway(settings).embed(texts, purpose=purpose, tenant_scope=tenant_scope)
    except EmbeddingUnavailable:
        raise
    except Exception as error:
        if not for_search:
            raise
        raise EmbeddingUnavailable("查询向量暂不可用，将使用关键词检索。") from error
