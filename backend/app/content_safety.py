"""输入、工具证据与模型输出共用的最小内容安全门。"""

from __future__ import annotations

import re

PROMPT_OVERRIDE = re.compile(
    r"(?:ignore|disregard|reveal|print).{0,48}(?:previous|system|prompt|instruction)|"
    r"(?:忽略|无视|绕过).{0,24}(?:之前|以上|系统|指令)|(?:系统提示词|开发者消息).{0,24}(?:输出|展示|泄露)",
    re.I,
)
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:api[_-]?key|secret|password|token)\s*[:=]\s*[^\s,;]+", re.I),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}\b", re.I),
)


def is_direct_prompt_override(text: str) -> bool:
    return bool(PROMPT_OVERRIDE.search(text))


def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def sanitize_untrusted_text(value: object, maximum: int = 12_000) -> str:
    text = str(value)[:maximum]
    text = PROMPT_OVERRIDE.sub("[已移除疑似提示词注入内容]", text)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[已隐藏敏感凭据]", text)
    return text
