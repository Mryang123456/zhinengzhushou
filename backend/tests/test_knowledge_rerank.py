"""验证 Cross-Encoder 与知识库检索的接入。"""

import os
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from backend.app.knowledge.embeddings import EmbeddingUnavailable
from backend.app.knowledge.store import KnowledgeStore, format_untrusted_evidence
from backend.evals.rag import measure


class KnowledgeRerankTests(unittest.TestCase):
    def setUp(self):
        self.store = KnowledgeStore.__new__(KnowledgeStore)
        self.store._connect = MagicMock()
        self.connection = self.store._connect.return_value
        self.cursor = self.connection.__enter__.return_value.cursor.return_value.__enter__.return_value
        seed_a = (7, "制度", 0, "年假适用于正式员工。", "范围 年假适用于正式员工。", "范围", ["范围"], 0, 1.0)
        seed_b = (8, "细则", 0, "年假申请规则。", "范围 年假申请规则。", "范围", ["范围"], 0, 0.8)
        neighbor = (8, "细则", 1, "试用期例外。", "试用期例外。", "范围", ["范围"], 0, 0.0)
        self.cursor.fetchall.side_effect = [[seed_a, seed_b], [seed_a, seed_b, neighbor]]
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, {
            "KNOWLEDGE_RERANK_MODEL": "test-model",
            "KNOWLEDGE_RERANK_TIMEOUT_SECONDS": "5",
            "KNOWLEDGE_NEIGHBOR_WINDOW": "1",
        }))
        stack.enter_context(patch("backend.app.knowledge.store.embed_texts", side_effect=EmbeddingUnavailable()))
        self.model = stack.enter_context(patch("backend.app.knowledge.cross_encoder._load_model")).return_value

    def test_second_pass_reorders_the_group_and_keeps_neighbor_citation(self):
        def predict(pairs, **kwargs):
            self.connection.__exit__.assert_called_once()
            if any("试用期例外" in passage for _, passage in pairs):
                return [0.2, 0.9]
            return [0.9, 0.7]

        self.model.predict.side_effect = predict
        stages = self.store.search_stages("年假", "owner", limit=1)
        self.assertEqual(self.model.predict.call_count, 2)
        self.assertEqual(stages.rerank_mode, "cross_encoder")
        self.assertEqual(stages.reranked[0]["citation"], "knowledge://7#0")
        self.assertEqual(stages.results[0]["citation"], "knowledge://8#0")
        self.assertIn("引用：knowledge://8#1", format_untrusted_evidence(stages.results[0]))
        self.assertEqual(measure(stages, {"knowledge://8#0", "knowledge://8#1"}, 1)["evidence_set_complete_at_k"], 1)

    def test_second_pass_failure_keeps_rule_order(self):
        self.model.predict.side_effect = [[0.1, 0.9], ValueError("模型不可用")]
        with self.assertLogs("backend.app.knowledge.cross_encoder", level="WARNING"):
            stages = self.store.search_stages("年假", "owner", limit=1)
        self.assertEqual(stages.rerank_mode, "fallback")
        self.assertEqual([row["citation"] for row in stages.results], ["knowledge://7#0"])

    def test_duplicate_windows_do_not_consume_top_k_in_either_mode(self):
        rows = [
            (7, "制度", 0, "年假规则。", "年假规则。", "范围", ["范围"], 0, 1.0),
            (7, "制度", 1, "年假例外。", "年假例外。", "范围", ["范围"], 0, 0.9),
            (8, "细则", 0, "年假定义。", "年假定义。", "范围", ["范围"], 0, 0.8),
        ]
        for model_name in ("", "test-model"):
            with self.subTest(model_name=model_name), patch.dict(os.environ, {"KNOWLEDGE_RERANK_MODEL": model_name}):
                self.cursor.fetchall.side_effect = [rows, rows[:2]]
                self.model.predict.reset_mock()
                self.model.predict.side_effect = [[0.9, 0.8, 0.7], [0.9, 0.8]]
                stages = self.store.search_stages("年假", "owner", limit=2)
                self.assertEqual(
                    [row["citations"] for row in stages.results],
                    [["knowledge://7#0", "knowledge://7#1"], ["knowledge://8#0"]],
                )
                if model_name:
                    self.assertEqual(len(self.model.predict.call_args.args[0]), 2)
