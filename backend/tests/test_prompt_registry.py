"""Prompt版本选择、完整性校验与Run固定版本。"""

import hashlib
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.app.prompts import prompt_manifest, prompt_manifest_hash, prompt_scope, prompt_text
from backend.app.prompts.registry import PromptRegistry, PromptRegistryError


def digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class PromptRegistryTests(unittest.TestCase):
    def test_application_manifest_is_complete_and_stable(self):
        manifest = prompt_manifest()
        self.assertGreaterEqual(len(manifest), 10)
        self.assertEqual([item["prompt_id"] for item in manifest], sorted(item["prompt_id"] for item in manifest))
        self.assertTrue(all(set(item) == {"prompt_id", "version", "hash", "owner", "change_reason"} for item in manifest))
        self.assertEqual(prompt_manifest_hash(manifest), prompt_manifest_hash(list(manifest)))

    def test_hash_mismatch_fails_closed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prompt.txt").write_text("实际内容", encoding="utf-8")
            (root / "registry.json").write_text(json.dumps({"prompts": {
                "demo": {"active": "1.0.0", "versions": {"1.0.0": {
                    "file": "prompt.txt", "sha256": digest("其他内容"),
                    "owner": "test", "change_reason": "test",
                }}},
            }}, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(PromptRegistryError):
                PromptRegistry(root / "registry.json")

    def test_environment_rollback_and_run_scope_pin_versions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "v1.txt").write_text("版本一", encoding="utf-8")
            (root / "v2.txt").write_text("版本二", encoding="utf-8")
            versions = {
                "1.0.0": {"file": "v1.txt", "sha256": digest("版本一"), "owner": "test", "change_reason": "baseline"},
                "2.0.0": {"file": "v2.txt", "sha256": digest("版本二"), "owner": "test", "change_reason": "upgrade"},
            }
            (root / "registry.json").write_text(json.dumps({"prompts": {
                "demo.prompt": {"active": "2.0.0", "versions": versions},
            }}, ensure_ascii=False), encoding="utf-8")
            registry = PromptRegistry(root / "registry.json")
            with patch.dict(os.environ, {"PROMPT_VERSION_DEMO_PROMPT": "1.0.0"}):
                manifest = registry.manifest()
                self.assertEqual(registry.get("demo.prompt").content, "版本一")
            with patch("backend.app.prompts.registry.get_registry", return_value=registry):
                with prompt_scope(manifest):
                    self.assertEqual(prompt_text("demo.prompt"), "版本一")
                self.assertEqual(prompt_text("demo.prompt"), "版本二")

    def test_incomplete_run_manifest_is_rejected(self):
        with self.assertRaises(PromptRegistryError):
            with prompt_scope(prompt_manifest()[:-1]):
                pass


if __name__ == "__main__":
    unittest.main()
