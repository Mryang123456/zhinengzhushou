"""从版本化资源加载 Prompt，并为每个 Run 固化可回放的版本清单。"""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from backend.app.config import BACKEND_ROOT


PROMPT_ROOT = Path(BACKEND_ROOT, "prompts")
REGISTRY_FILE = PROMPT_ROOT / "registry.json"
_selected_versions: ContextVar[dict[str, str] | None] = ContextVar("prompt_versions", default=None)
_registry = None
_registry_lock = Lock()


class PromptRegistryError(RuntimeError):
    """Prompt资源缺失、Hash不匹配或版本不可用。"""


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    version: str
    sha256: str
    owner: str
    change_reason: str
    content: str

    def manifest_entry(self) -> dict[str, str]:
        return {
            "prompt_id": self.prompt_id,
            "version": self.version,
            "hash": self.sha256,
            "owner": self.owner,
            "change_reason": self.change_reason,
        }


class PromptRegistry:
    def __init__(self, registry_file: Path = REGISTRY_FILE):
        self.registry_file = registry_file
        self.root = registry_file.parent
        try:
            payload = json.loads(registry_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PromptRegistryError(f"无法读取Prompt Registry：{registry_file}") from error
        prompts = payload.get("prompts")
        if not isinstance(prompts, dict) or not prompts:
            raise PromptRegistryError("Prompt Registry没有有效的prompts配置。")
        self._config = prompts
        self._records: dict[tuple[str, str], PromptRecord] = {}
        self._load_all()

    def _load_all(self) -> None:
        for prompt_id, config in self._config.items():
            versions = config.get("versions", {})
            active = config.get("active")
            if active not in versions:
                raise PromptRegistryError(f"Prompt {prompt_id} 的active版本不存在。")
            for version, metadata in versions.items():
                file_path = (self.root / metadata["file"]).resolve()
                if self.root.resolve() not in file_path.parents:
                    raise PromptRegistryError(f"Prompt {prompt_id} 的文件越出Registry目录。")
                try:
                    content = file_path.read_text(encoding="utf-8").rstrip("\r\n")
                except OSError as error:
                    raise PromptRegistryError(f"Prompt文件不存在：{file_path}") from error
                actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
                expected = str(metadata.get("sha256", "")).lower()
                if not expected or actual != expected:
                    raise PromptRegistryError(
                        f"Prompt {prompt_id}@{version} Hash不匹配，禁止启动。"
                    )
                owner = str(metadata.get("owner", "")).strip()
                change_reason = str(metadata.get("change_reason", "")).strip()
                if not owner or not change_reason:
                    raise PromptRegistryError(f"Prompt {prompt_id}@{version} 缺少owner或change_reason。")
                self._records[(prompt_id, version)] = PromptRecord(
                    prompt_id=prompt_id,
                    version=version,
                    sha256=actual,
                    owner=owner,
                    change_reason=change_reason,
                    content=content,
                )

    @staticmethod
    def _version_env(prompt_id: str) -> str:
        suffix = re.sub(r"[^A-Z0-9]+", "_", prompt_id.upper()).strip("_")
        return f"PROMPT_VERSION_{suffix}"

    def selected_version(self, prompt_id: str) -> str:
        pinned = _selected_versions.get()
        if pinned and prompt_id in pinned:
            return pinned[prompt_id]
        configured = os.getenv(self._version_env(prompt_id), "").strip()
        return configured or str(self._config[prompt_id]["active"])

    def get(self, prompt_id: str) -> PromptRecord:
        if prompt_id not in self._config:
            raise PromptRegistryError(f"未注册Prompt：{prompt_id}")
        version = self.selected_version(prompt_id)
        try:
            return self._records[(prompt_id, version)]
        except KeyError as error:
            raise PromptRegistryError(f"Prompt版本不可用：{prompt_id}@{version}") from error

    def manifest(self) -> list[dict[str, str]]:
        return [self.get(prompt_id).manifest_entry() for prompt_id in sorted(self._config)]

    def validate_manifest(self, manifest: list[dict]) -> dict[str, str]:
        if not isinstance(manifest, list) or len(manifest) != len(self._config):
            raise PromptRegistryError("Run的Prompt Manifest数量不正确，禁止静默续跑。")
        selected: dict[str, str] = {}
        for item in manifest:
            prompt_id, version = str(item.get("prompt_id", "")), str(item.get("version", ""))
            record = self._records.get((prompt_id, version))
            if record is None or item.get("hash") != record.sha256:
                raise PromptRegistryError(f"Run引用了不可回放的Prompt：{prompt_id}@{version}")
            selected[prompt_id] = version
        expected = set(self._config)
        if set(selected) != expected:
            raise PromptRegistryError("Run的Prompt Manifest不完整，禁止使用当前版本静默续跑。")
        return selected


def get_registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = PromptRegistry()
    return _registry


def prompt_text(prompt_id: str) -> str:
    return get_registry().get(prompt_id).content


def prompt_manifest() -> list[dict[str, str]]:
    return get_registry().manifest()


def prompt_manifest_hash(manifest: list[dict] | None = None) -> str:
    canonical = json.dumps(prompt_manifest() if manifest is None else manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@contextmanager
def prompt_scope(manifest: list[dict]):
    """将Run固定到提交时的Prompt版本，恢复时不跟随active版本漂移。"""
    selected = get_registry().validate_manifest(manifest)
    token = _selected_versions.set(selected)
    try:
        yield
    finally:
        _selected_versions.reset(token)
