"""独立工具检索层：语义召回候选，绝不替代后续权限白名单。"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolDocument:
    name: str
    description: str
    tags: tuple[str, ...]
    schema_hint: str

    @property
    def text(self) -> str:
        return f"工具：{self.name}\n说明：{self.description}\n标签：{'、'.join(self.tags)}\n参数：{self.schema_hint}"


def infer_tags(name: str, description: str) -> tuple[str, ...]:
    text = f"{name} {description}".lower()
    mapping = {"天气": ("天气", "气温", "地点"), "搜索": ("搜索", "检索", "网页", "新闻"), "memory": ("记忆", "保存", "删除"), "calculator": ("计算", "数学"), "knowledge": ("知识库", "文档", "检索")}
    return tuple(tag for tag, hints in mapping.items() if any(hint in text for hint in hints)) or ("通用",)


def build_tool_documents(tools: dict[str, object]) -> list[ToolDocument]:
    documents = []
    for name, tool in tools.items():
        description = str(getattr(tool, "description", ""))
        args_schema = getattr(tool, "args_schema", None)
        schema_hint = str(getattr(args_schema, "model_fields", ""))[:600]
        documents.append(ToolDocument(name, description, infer_tags(name, description), schema_hint))
    return documents


def _terms(text: str) -> set[str]:
    normalized = text.lower()
    words = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    for run in re.findall(r"[\u4e00-\u9fff]+", normalized):
        words.update(run[index:index + 2] for index in range(max(0, len(run) - 1)))
    return words


def _lexical_rank(question: str, documents: list[ToolDocument], limit: int) -> list[str]:
    terms = _terms(question)
    return [item.name for item in sorted(documents, key=lambda item: (len(terms & _terms(item.text)), item.name), reverse=True)[:limit]]


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator if denominator else 0.0


def retrieve_tool_candidates(question: str, tools: dict[str, object], limit: int = 12) -> tuple[list[str], str]:
    """显式启用 OpenAI embedding 时做向量召回，否则确定性词面降级。"""
    documents = build_tool_documents(tools)
    limit = max(1, min(limit, len(documents)))
    if os.getenv("TOOL_RETRIEVAL_EMBEDDING_PROVIDER", "").lower() != "openai":
        return _lexical_rank(question, documents, limit), "lexical"
    try:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model=os.getenv("TOOL_RETRIEVAL_EMBEDDING_MODEL", "text-embedding-3-small"))
        vectors = embeddings.embed_documents([document.text for document in documents])
        query_vector = embeddings.embed_query(question)
        ranked = sorted(zip(documents, vectors, strict=True), key=lambda item: (_cosine(query_vector, item[1]), item[0].name), reverse=True)
        return [document.name for document, _ in ranked[:limit]], "vector"
    except Exception:
        return _lexical_rank(question, documents, limit), "lexical_fallback"
