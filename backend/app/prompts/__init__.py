"""版本化 Prompt Registry。"""

from backend.app.prompts.registry import (
    PromptRegistryError,
    prompt_manifest,
    prompt_manifest_hash,
    prompt_scope,
    prompt_text,
)

__all__ = [
    "PromptRegistryError",
    "prompt_manifest",
    "prompt_manifest_hash",
    "prompt_scope",
    "prompt_text",
]
