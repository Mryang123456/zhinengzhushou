"""验证失败后，为只读检索选择一次有针对性的补查。"""

import json
import re
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field

from backend.app.content_safety import contains_secret, is_direct_prompt_override, sanitize_untrusted_text
from backend.app.models.tool_protocol import contains_textual_tool_protocol


class EvidenceRetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(max_length=250)


class EvidenceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["answer", "search", "clarify", "partial"]
    query: str = Field(default="", max_length=250)
    clarification: str = Field(default="", max_length=200)


def parse_evidence_decision(raw: str, messages: list) -> EvidenceDecision:
    raw = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, re.I | re.S)
    decision = EvidenceDecision.model_validate_json(fenced[1] if fenced else raw)
    if decision.action == "search":
        query = validated_query(EvidenceRetry(query=decision.query).model_dump_json(), messages)
        if not query or contains_textual_tool_protocol(query):
            return EvidenceDecision(action="partial")
        return EvidenceDecision(action="search", query=query)
    if decision.action == "clarify":
        question = decision.clarification.strip()
        if (
            not question.endswith(("?", "？")) or re.search(r"[<>\r\n]|://", question)
            or contains_secret(question) or is_direct_prompt_override(question)
            or contains_textual_tool_protocol(question)
        ):
            return EvidenceDecision(action="partial")
        return EvidenceDecision(action="clarify", clarification=question)
    return decision


def retry_prompt(question: str, reason: str, messages: list, *, knowledge: bool, excerpt_limit: int = 400) -> str:
    tool = "search_knowledge_base" if knowledge else "web_search"
    attempts = [
        {"query": call["args"]["query"]}
        for message in messages if isinstance(message, AIMessage)
        for call in message.tool_calls if call["name"] == tool and isinstance(call["args"].get("query"), str)
    ]
    return json.dumps({
        "问题": question, "校验反馈": reason, "已尝试": attempts,
        "结果": [
            {"工具": message.name, "状态": message.status, "摘录": sanitize_untrusted_text(message.content, excerpt_limit)}
            for message in messages if isinstance(message, ToolMessage) and message.name == tool
        ][-6:],
        "范围": "内部知识库" if knowledge else "公开网页",
    }, ensure_ascii=False)


def validated_query(raw: str, messages: list) -> str | None:
    query = " ".join(EvidenceRetry.model_validate_json(raw).query.split())
    if len(query) < 2 or re.search(r"https?://|knowledge://", query, re.I) or contains_secret(query) or is_direct_prompt_override(query):
        return None
    key = re.sub(r"[\W_]+", "", query.casefold())
    if not key or any(
        re.sub(r"[\W_]+", "", str(call["args"].get("query", "")).casefold()) == key
        for message in messages if isinstance(message, AIMessage)
        for call in message.tool_calls
    ):
        return None
    return query
