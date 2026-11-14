"""兼容模型误输出到正文中的文本工具协议，并阻止协议原文泄漏。"""

from __future__ import annotations

import html
import json
import re
from uuid import uuid4


_DSML_MARKER = re.compile(r"\|+\s*DSML\s*\|+", re.I)
_CALLS = re.compile(r"^<\|DSML\|calls>(.*?)</\|DSML\|calls>$", re.I | re.S)
_INVOKE = re.compile(
    r'<\|DSML\|invoke\s+name=["\']([^"\']+)["\']\s*>(.*?)</\|DSML\|invoke>',
    re.I | re.S,
)
_PARAMETER = re.compile(
    r'<\|DSML\|parameter\s+name=["\']([^"\']+)["\'](?:\s+[^>]*)?>(.*?)</\|DSML\|parameter>',
    re.I | re.S,
)


def contains_textual_tool_protocol(text: str) -> bool:
    """识别 DSML 等不应展示给最终用户的模型内部协议。"""
    return bool(re.search(r"DSML\s*[|｜]", text, re.I) or re.search(r"[|｜]\s*DSML", text, re.I))


def _normalise_dsml(text: str) -> str:
    compact = text.replace("｜", "|").replace("¦", "|")
    compact = _DSML_MARKER.sub("|DSML|", compact)
    compact = re.sub(r"<\s*(/?)\s*\|DSML\|\s*", r"<\1|DSML|", compact, flags=re.I)
    compact = re.sub(r"\s+>", ">", compact)
    return re.sub(r">\s+<", "><", compact.strip())


def _parameter_value(value: str):
    value = html.unescape(value.strip())
    if not value:
        return ""
    if value[0] in "[{" or value in {"true", "false", "null"} or re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def parse_textual_tool_calls(text: str, allowed_names: set[str]) -> list[dict] | None:
    """把完整、可验证且仅包含白名单工具的 DSML 正文转换为标准 tool_calls。

    解析采用 fail-closed：存在包装外文本、未知工具、重复参数或残余标记时均拒绝。
    """
    if not contains_textual_tool_protocol(text):
        return None
    normalised = _normalise_dsml(text)
    wrapper = _CALLS.fullmatch(normalised)
    if not wrapper:
        return None
    body = wrapper.group(1)
    matches = list(_INVOKE.finditer(body))
    if not matches or "".join(match.group(0) for match in matches) != body:
        return None

    calls = []
    for match in matches:
        name, parameter_body = match.groups()
        if name not in allowed_names:
            return None
        parameters = list(_PARAMETER.finditer(parameter_body))
        if "".join(parameter.group(0) for parameter in parameters) != parameter_body:
            return None
        arguments = {}
        for parameter in parameters:
            key, raw_value = parameter.groups()
            if key in arguments:
                return None
            arguments[key] = _parameter_value(raw_value)
        calls.append({
            "name": name,
            "args": arguments,
            "id": f"protocol-{uuid4().hex}",
            "type": "tool_call",
        })
    return calls
