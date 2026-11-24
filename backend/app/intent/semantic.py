"""用本轮请求和近期对话解析意图，不授予工具权限。"""

import asyncio
import json
import logging
import re
from datetime import date
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from backend.app.content_safety import contains_secret, is_direct_prompt_override, sanitize_untrusted_text
from backend.app.prompts import prompt_text

logger = logging.getLogger(__name__)
NO_WEB = re.compile(r"(?:不要|不用|无需|禁止|别|不必|不能|不)\s*(?:再|去)?\s*(?:联网|上网|搜索|搜网页|查网页)")
NO_WEB_CLAUSE = re.compile(rf"(?:^|(?<=[，,。；;]))\s*(?:请\s*)?{NO_WEB.pattern}\s*(?=$|[，,。；;！!？?])")


def without_no_web_clause(question: str) -> str:
    """提供给规则分类器的文本仍保留问句本身的标点。"""
    return NO_WEB_CLAUSE.sub("", question).strip(" ，,。；;！!")


def is_public_search_tool(name: str) -> bool:
    return name == "web_search" or (name.startswith("search_") and name != "search_knowledge_base")


class SemanticIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Literal["direct", "search", "tools", "clarify"]
    question: str = Field(min_length=1, max_length=1500)
    query: str = Field(default="", max_length=250)
    queries: list[str] = Field(default_factory=list, max_length=2)
    clarification: str = Field(default="", max_length=200)


async def resolve_intent(model, question: str, history: list, *, capabilities: dict | None = None) -> SemanticIntent | None:
    invoke = getattr(model, "ainvoke", None)
    if not callable(invoke) or len(question) > 1500 or contains_secret(question):
        return None
    context = [
        {"role": "用户" if isinstance(message, HumanMessage) else "助手",
         "text": sanitize_untrusted_text(message.content, 700)}
        for message in history
        if isinstance(message, HumanMessage) or (isinstance(message, AIMessage) and not message.tool_calls)
    ][-10:]
    instruction = SystemMessage(
        prompt_text("intent.semantic")
        + "JSON Schema：" + json.dumps(SemanticIntent.model_json_schema(), ensure_ascii=False)
    )
    try:
        async with asyncio.timeout(6):
            response = await invoke([instruction, HumanMessage(json.dumps({
                "当前请求": question, "近期对话": context, "当前日期": date.today().isoformat(),
                "后端能力": capabilities,
            }, ensure_ascii=False))])
        if not isinstance(response.content, str):
            return None
        raw = response.content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, re.I | re.S)
        decision = SemanticIntent.model_validate_json(fenced[1] if fenced else raw)
    except Exception as error:
        logger.info("语义意图识别回退：%s", type(error).__name__)
        return None
    for value in (decision.question, decision.query, decision.clarification, *decision.queries):
        if contains_secret(value) or is_direct_prompt_override(value) or "<" in value or ">" in value:
            return None
    if not decision.question.strip():
        return None
    if decision.route == "search" and (not decision.query.strip() or "://" in decision.query):
        return None
    if any(not value.strip() or "://" in value for value in decision.queries):
        return None
    if decision.route == "clarify" and not decision.clarification.strip():
        return None
    return decision
