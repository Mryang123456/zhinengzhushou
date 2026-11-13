"""Embedding Gateway 的缓存、切批、租户隔离和熔断测试。"""

from dataclasses import replace
from unittest.mock import MagicMock
from unittest.mock import patch
import unittest

from fastapi.testclient import TestClient

from backend.app.knowledge.embedding.cache import cache_key
from backend.app.knowledge.embedding.config import EmbeddingSettings
from backend.app.knowledge.embedding.gateway import EmbeddingGateway
from backend.app.knowledge.embedding.resilience import EmbeddingCircuitOpen


def settings(**overrides) -> EmbeddingSettings:
    base = EmbeddingSettings(
        provider="openai", model="test-model", model_revision="v1", dimensions=2,
        service_url="", service_api_key="", timeout_seconds=0.1,
        max_batch_items=2, max_batch_tokens=100, max_text_tokens=100,
        max_concurrency=1, requests_per_minute=100, tokens_per_minute=10000,
        tenant_requests_per_minute=50, tenant_tokens_per_minute=5000,
        retry_attempts=1, cache_ttl_seconds=60, cache_max_entries=100,
        normalization_version="nfc-v1", breaker_failures=2,
        breaker_recovery_seconds=60, redis_url="",
    )
    return replace(base, **overrides)


class FakeProvider:
    def __init__(self):
        self.calls = []

    def embed(self, texts, purpose, tenant_scope):
        self.calls.append((list(texts), purpose, tenant_scope))
        return [[float(len(text)), 1.0] for text in texts]


class EmbeddingGatewayTests(unittest.TestCase):
    def test_deduplicates_batches_and_reuses_content_addressed_cache(self):
        provider = FakeProvider()
        gateway = EmbeddingGateway(settings(), provider)
        result = gateway.embed(["甲", "乙", "甲", "丙"], purpose="document", tenant_scope="tenant-a")
        self.assertEqual(result, [[1.0, 1.0]] * 4)
        self.assertEqual([call[0] for call in provider.calls], [["甲", "乙"], ["丙"]])
        gateway.embed(["甲", "乙"], purpose="document", tenant_scope="tenant-a")
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(gateway.invalidate_scope("tenant-a"), 3)
        gateway.embed(["甲"], purpose="document", tenant_scope="tenant-a")
        self.assertEqual(len(provider.calls), 3)

    def test_cache_key_isolated_by_tenant_purpose_and_model_revision(self):
        base = settings()
        key = cache_key(base, "制度", "document", "tenant-a")
        self.assertNotEqual(key, cache_key(base, "制度", "document", "tenant-b"))
        self.assertNotEqual(key, cache_key(base, "制度", "query", "tenant-a"))
        self.assertNotEqual(key, cache_key(replace(base, model_revision="v2"), "制度", "document", "tenant-a"))
        self.assertEqual(key, cache_key(base, "制度\r\n", "document", "tenant-a"))

    def test_transient_failures_open_circuit_and_skip_provider(self):
        provider = MagicMock()
        provider.embed.side_effect = ConnectionError("down")
        gateway = EmbeddingGateway(settings(), provider)
        for text in ("甲", "乙"):
            with self.assertRaises(ConnectionError):
                gateway.embed([text], purpose="query", tenant_scope="tenant-a")
        with self.assertRaises(EmbeddingCircuitOpen):
            gateway.embed(["丙"], purpose="query", tenant_scope="tenant-a")
        self.assertEqual(provider.embed.call_count, 2)

    def test_rejects_oversized_text_before_calling_provider(self):
        provider = MagicMock()
        gateway = EmbeddingGateway(settings(max_text_tokens=1), provider)
        with self.assertRaises(ValueError):
            gateway.embed(["过长文本"], purpose="document", tenant_scope="tenant-a")
        provider.embed.assert_not_called()

    def test_standalone_gateway_requires_authentication_and_returns_contract(self):
        from backend.app.knowledge.embedding import api

        client = TestClient(api.app)
        with patch.dict("os.environ", {"EMBEDDING_SERVICE_API_KEY": "test-key"}):
            self.assertEqual(client.post(
                "/v1/embeddings", json={"texts": ["制度"], "purpose": "document"},
            ).status_code, 401)
            with (
                patch.object(api, "settings", replace(api.settings, provider="openai")),
                patch.object(api.gateways["document"], "embed", return_value=[[0.0] * 1536]),
            ):
                response = client.post(
                    "/v1/embeddings",
                    headers={"Authorization": "Bearer test-key"},
                    json={"texts": ["制度"], "purpose": "document", "tenant_scope": "tenant-a"},
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["dimensions"], 1536)
        self.assertEqual(len(response.json()["vectors"][0]), 1536)


if __name__ == "__main__":
    unittest.main()
