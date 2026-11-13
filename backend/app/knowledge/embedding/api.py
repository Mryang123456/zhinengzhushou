"""独立 Embedding Gateway 服务：uvicorn backend.app.knowledge.embedding.api:app。"""

from __future__ import annotations

import hmac
import os

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from backend.app.knowledge.embedding.config import EmbeddingSettings
from backend.app.knowledge.embedding.gateway import EmbeddingGateway
from backend.app.knowledge.embedding.providers import OpenAIEmbeddingProvider
from backend.app.knowledge.embedding.resilience import EmbeddingCircuitOpen, EmbeddingRateLimited
from backend.app.observability import prometheus_metrics


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=2048)
    purpose: str = Field(pattern="^(query|document)$")
    tenant_scope: str = Field(default="service", min_length=1, max_length=128)


class EmbeddingResponse(BaseModel):
    model: str
    model_revision: str
    dimensions: int
    vectors: list[list[float]]


class CacheInvalidationRequest(BaseModel):
    tenant_scope: str = Field(min_length=1, max_length=128)


settings = EmbeddingSettings.from_env()
if os.getenv("ENVIRONMENT", "development").lower() in {"production", "prod"}:
    missing = [name for name in ("EMBEDDING_SERVICE_API_KEY", "REDIS_URL", "KNOWLEDGE_EMBEDDING_MODEL_REVISION") if not os.getenv(name)]
    if settings.provider == "disabled":
        missing.append("KNOWLEDGE_EMBEDDING_PROVIDER")
    if missing:
        raise RuntimeError(f"生产环境独立 Embedding 服务缺少配置：{', '.join(missing)}")
    if len(os.getenv("EMBEDDING_SERVICE_API_KEY", "")) < 32:
        raise RuntimeError("生产环境 EMBEDDING_SERVICE_API_KEY 至少需要 32 个字符。")
gateways = {
    purpose: EmbeddingGateway(settings.for_purpose(purpose), OpenAIEmbeddingProvider(settings.for_purpose(purpose)))
    for purpose in ("query", "document")
}
app = FastAPI(title="Embedding Gateway", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "provider": settings.provider, "model": settings.model}


@app.get("/ready")
def ready() -> dict:
    if settings.provider == "disabled":
        raise HTTPException(status_code=503, detail="Embedding Provider 未启用。")
    redis_client = gateways["document"].cache.redis
    if redis_client is not None:
        try:
            redis_client.ping()
        except Exception as error:
            raise HTTPException(status_code=503, detail="Embedding Redis 不可用。") from error
    return {"status": "ready"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(prometheus_metrics(), media_type="text/plain; version=0.0.4")


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def embeddings(request: EmbeddingRequest, authorization: str | None = Header(default=None)) -> EmbeddingResponse:
    expected = os.getenv("EMBEDDING_SERVICE_API_KEY", "")
    provided = (authorization or "").removeprefix("Bearer ")
    if not expected or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Embedding 服务认证失败。")
    if settings.provider == "disabled":
        raise HTTPException(status_code=503, detail="Embedding Provider 未启用。")
    try:
        vectors = gateways[request.purpose].embed(
            request.texts, purpose=request.purpose, tenant_scope=request.tenant_scope,
        )
    except EmbeddingRateLimited as error:
        raise HTTPException(status_code=429, detail=str(error), headers={"Retry-After": "60"}) from error
    except EmbeddingCircuitOpen as error:
        raise HTTPException(status_code=503, detail=str(error), headers={"Retry-After": "30"}) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="Embedding 服务暂时不可用。") from error
    return EmbeddingResponse(
        model=settings.model, model_revision=settings.model_revision,
        dimensions=settings.dimensions, vectors=vectors,
    )


@app.post("/v1/cache/invalidate")
def invalidate_cache(request: CacheInvalidationRequest, authorization: str | None = Header(default=None)) -> dict:
    expected = os.getenv("EMBEDDING_SERVICE_API_KEY", "")
    provided = (authorization or "").removeprefix("Bearer ")
    if not expected or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Embedding 服务认证失败。")
    return {"invalidated": sum(gateway.invalidate_scope(request.tenant_scope) for gateway in gateways.values())}
