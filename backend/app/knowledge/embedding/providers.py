"""本地 Provider 和独立 Embedding 服务客户端。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol

from backend.app.knowledge.embedding.config import EmbeddingSettings


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str], purpose: str, tenant_scope: str) -> list[list[float]]: ...


class OpenAIEmbeddingProvider:
    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings

    def embed(self, texts: list[str], purpose: str, tenant_scope: str) -> list[list[float]]:
        from langchain_openai import OpenAIEmbeddings
        model = OpenAIEmbeddings(
            model=self.settings.model,
            dimensions=self.settings.dimensions,
            request_timeout=self.settings.timeout_seconds,
            max_retries=0,
        )
        try:
            return model.embed_documents(texts)
        except TimeoutError:
            raise
        except Exception as error:
            status = getattr(error, "status_code", None)
            if isinstance(status, int) and status < 500 and status != 429:
                raise ValueError("Embedding Provider 拒绝了请求。") from error
            raise ConnectionError("Embedding Provider 暂时不可用。") from error


class RemoteEmbeddingProvider:
    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings

    def embed(self, texts: list[str], purpose: str, tenant_scope: str) -> list[list[float]]:
        body = json.dumps({"texts": texts, "purpose": purpose, "tenant_scope": tenant_scope}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.settings.service_api_key:
            headers["Authorization"] = f"Bearer {self.settings.service_api_key}"
        request = urllib.request.Request(
            f"{self.settings.service_url}/v1/embeddings", data=body, headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code < 500 and error.code != 429:
                raise ValueError(f"Embedding 服务拒绝请求，HTTP {error.code}。") from error
            raise ConnectionError(f"Embedding 服务暂时不可用，HTTP {error.code}。") from error
        vectors = payload.get("vectors")
        if not isinstance(vectors, list):
            raise ValueError("Embedding 服务响应格式无效。")
        if (
            payload.get("model") != self.settings.model
            or payload.get("model_revision") != self.settings.model_revision
            or payload.get("dimensions") != self.settings.dimensions
        ):
            raise ValueError("Embedding 服务模型契约与调用方配置不一致。")
        return vectors

    def invalidate_scope(self, tenant_scope: str) -> int:
        body = json.dumps({"tenant_scope": tenant_scope}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.settings.service_url}/v1/cache/invalidate", data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.service_api_key}",
            }, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise ConnectionError(f"Embedding 缓存失效请求失败，HTTP {error.code}。") from error
        return int(payload.get("invalidated", 0))
