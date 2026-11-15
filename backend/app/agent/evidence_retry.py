"""验证失败后，为只读检索选择一次有针对性的补查。"""

import json
import re
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field

from backend.app.content_safety import contains_secret, is_direct_prompt_override, sanitize_untrusted_text
from backend.app.intent.semantic import is_public_search_tool
from backend.app.models.tool_protocol import contains_textual_tool_protocol
from backend.app.quality.validator import web_answer_sources
from backend.app.quality.web_writing import numbered_web_excerpts


def is_evidence_tool(name: str | None, *, knowledge: bool) -> bool:
    return bool(name and (name == "search_knowledge_base" if knowledge else is_public_search_tool(name)))


class EvidenceRetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(max_length=250)


class EvidenceRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cells: list[str] = Field(min_length=2, max_length=4)
    sources: list[int] = Field(min_length=1, max_length=3)


class EvidenceTable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="", max_length=50)
    columns: list[str] = Field(min_length=2, max_length=4)
    rows: list[EvidenceRow] = Field(min_length=2, max_length=10)


class EvidenceDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=32)
    text: str = Field(min_length=1, max_length=200)
    sources: list[int] = Field(min_length=1, max_length=3)


class EvidenceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["answer", "search", "clarify", "partial"]
    query: str = Field(default="", max_length=250)
    clarification: str = Field(default="", max_length=200)
    missing: str = Field(default="", max_length=200)
    table: EvidenceTable | None = None
    details: list[EvidenceDetail] = Field(default_factory=list, max_length=3)


def parse_evidence_decision(raw: str, messages: list) -> EvidenceDecision:
    raw = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, re.I | re.S)
    payload = json.loads(fenced[1] if fenced else raw)
    if not isinstance(payload, dict):
        raise ValueError("证据决策应为对象")
    table_data = payload.pop("table", None)
    details_data = payload.pop("details", None)
    decision = EvidenceDecision.model_validate(payload)
    if table_data is not None and decision.action in {"answer", "partial"}:
        try:
            decision = decision.model_copy(update={"table": EvidenceTable.model_validate(table_data)})
        except ValueError:
            pass  # 可选表格有结构问题时仍继续用正文作答。
    if isinstance(details_data, list) and decision.action in {"answer", "partial"}:
        details = []
        for item in details_data[:3]:
            try:
                details.append(EvidenceDetail.model_validate(item))
            except ValueError:
                continue
        decision = decision.model_copy(update={"details": details})
    missing = sanitize_untrusted_text(decision.missing, 200).strip()
    if contains_secret(missing) or is_direct_prompt_override(missing) or contains_textual_tool_protocol(missing) or "://" in missing:
        missing = ""
    if decision.action == "search":
        query = validated_query(EvidenceRetry(query=decision.query).model_dump_json(), messages)
        if not query or contains_textual_tool_protocol(query):
            return EvidenceDecision(action="partial", missing=missing)
        return EvidenceDecision(action="search", query=query, missing=missing)
    if decision.action == "clarify":
        question = decision.clarification.strip()
        if (
            not question.endswith(("?", "？")) or re.search(r"[<>\r\n]|://", question)
            or contains_secret(question) or is_direct_prompt_override(question)
            or contains_textual_tool_protocol(question)
        ):
            return EvidenceDecision(action="partial")
        return EvidenceDecision(action="clarify", clarification=question)
    return decision.model_copy(update={"missing": missing})


def retry_prompt(question: str, reason: str, messages: list, *, knowledge: bool, excerpt_limit: int = 400) -> str:
    attempts = [
        {"query": call["args"]["query"]}
        for message in messages if isinstance(message, AIMessage)
        for call in message.tool_calls if is_evidence_tool(call["name"], knowledge=knowledge) and isinstance(call["args"].get("query"), str)
    ]
    sources = [] if knowledge else web_answer_sources(question, messages)
    results = numbered_web_excerpts(sources) if sources else [
        {"工具": message.name, "状态": message.status, "摘录": sanitize_untrusted_text(message.content, excerpt_limit)}
        for message in messages if isinstance(message, ToolMessage) and is_evidence_tool(message.name, knowledge=knowledge)
    ][-6:]
    return json.dumps({
        "问题": question, "校验反馈": reason, "已尝试": attempts, "结果": results,
        "范围": "内部知识库" if knowledge else "公开网页",
    }, ensure_ascii=False)


def validated_query(raw: str, messages: list) -> str | None:
    raw = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, re.I | re.S)
    query = " ".join(EvidenceRetry.model_validate_json(fenced[1] if fenced else raw).query.split())
    if (len(query) < 2 or "://" in query or contains_secret(query) or is_direct_prompt_override(query)
            or contains_textual_tool_protocol(query)):
        return None
    key = re.sub(r"[\W_]+", "", query.casefold())
    if not key or any(
        re.sub(r"[\W_]+", "", str(call["args"].get("query", "")).casefold()) == key
        for message in messages if isinstance(message, AIMessage)
        for call in message.tool_calls
    ):
        return None
    return query
