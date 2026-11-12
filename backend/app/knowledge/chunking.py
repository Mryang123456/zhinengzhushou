"""结构感知的确定性文档切分。

检索使用较小的语义块；标题路径作为上下文参与向量化，但原始正文单独保留，
以便回答时能够逐字引用。旧 ``split_text`` API 保留给兼容调用方。
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass


CHUNKER_VERSION = "structure-v4"
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_CHAPTER_HEADING = re.compile(r"^第[一二三四五六七八九十百千万0-9]+[章节篇部]\s*.*$")
_SENTENCE_END = re.compile(r"(?<=[。！？；])|(?<=[.!?;])(?=\s)")
_LIST_ITEM = re.compile(r"^(?:[-*+]\s+|\d+[.)、]\s*|[（(]?[一二三四五六七八九十]+[）)、.]\s*)")


@dataclass(frozen=True)
class DocumentChunk:
    """可持久化的知识片段及其检索上下文。"""

    index: int
    content: str
    contextual_content: str
    section_title: str
    section_path: tuple[str, ...]
    section_index: int
    token_count: int
    content_hash: str


def estimate_tokens(text: str) -> int:
    """保守估算中英文混合文本 token 数，不绑定具体模型 tokenizer。"""
    total = 0
    for token in _TOKEN_RE.findall(text):
        total += math.ceil(len(token) / 4) if token.isascii() and token.isalnum() else 1
    return total


def chunking_fingerprint(max_tokens: int, overlap_tokens: int) -> str:
    return f"{CHUNKER_VERSION}:{max_tokens}:{overlap_tokens}"


def _heading(line: str) -> tuple[int, str] | None:
    if match := _MARKDOWN_HEADING.match(line):
        return len(match.group(1)), match.group(2).strip()
    if _CHAPTER_HEADING.match(line):
        return 1, line.strip()
    return None


def _sections(text: str) -> list[tuple[tuple[str, ...], list[str]]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u200b", "")
    headings: list[str] = []
    current_path: tuple[str, ...] = ()
    paragraph: list[str] = []
    sections: list[tuple[tuple[str, ...], list[str]]] = []
    blocks: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            value = "\n".join(part.strip() for part in paragraph if part.strip()).strip()
            if value:
                blocks.append(value)
            paragraph.clear()

    def flush_section() -> None:
        flush_paragraph()
        if blocks:
            sections.append((current_path, blocks.copy()))
            blocks.clear()

    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue
        detected = _heading(line)
        if detected:
            flush_section()
            level, title = detected
            headings[level - 1 :] = []
            while len(headings) < level - 1:
                headings.append("")
            headings.append(title)
            current_path = tuple(item for item in headings if item)
            continue
        if "|" in line or _LIST_ITEM.match(line):
            # 列表项和表格行是不可拆散的业务语义单元。
            flush_paragraph()
            blocks.append(line)
            continue
        paragraph.append(line)
    flush_section()
    return sections


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """先按句子切；单句仍过大时按字符窗口确定性切分。"""
    if estimate_tokens(text) <= max_tokens:
        return [text]
    sentences = [item for item in _SENTENCE_END.split(text) if item.strip()]
    if not sentences:
        sentences = [text]
    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if estimate_tokens(sentence) > max_tokens:
            if current:
                pieces.append("".join(current))
                current = []
            start = 0
            while start < len(sentence):
                end = min(len(sentence), start + max_tokens)
                while end > start + 1 and estimate_tokens(sentence[start:end]) > max_tokens:
                    end -= max(1, (end - start) // 8)
                pieces.append(sentence[start:end].strip())
                start = end
            continue
        if current and estimate_tokens("".join(current) + sentence) > max_tokens:
            pieces.append("".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append("".join(current))
    return [piece.strip() for piece in pieces if piece.strip()]


def _tail_within_budget(text: str, budget: int) -> str:
    if budget <= 0 or not text:
        return ""
    tokens = list(_TOKEN_RE.finditer(text))
    used = 0
    start = len(text)
    for match in reversed(tokens):
        cost = estimate_tokens(match.group())
        if used + cost > budget:
            break
        used += cost
        start = match.start()
    return text[start:].strip()


def chunk_document(
    text: str,
    *,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[DocumentChunk]:
    """按章节和语义块切分文档，且只在同一章节内建立重叠窗口。"""
    maximum = max_tokens or int(os.getenv("KNOWLEDGE_CHUNK_TOKENS", "500"))
    overlap = overlap_tokens if overlap_tokens is not None else int(os.getenv("KNOWLEDGE_CHUNK_OVERLAP_TOKENS", "80"))
    if maximum < 64 or overlap < 0 or overlap >= maximum // 2:
        raise ValueError("知识切片要求 max_tokens >= 64，且 overlap_tokens 小于 max_tokens 的一半。")

    chunks: list[DocumentChunk] = []
    for section_index, (path, blocks) in enumerate(_sections(text)):
        pieces = [piece for block in blocks for piece in _hard_split(block, maximum)]
        current: list[str] = []

        def emit() -> str:
            content = "\n\n".join(current).strip()
            if not content:
                return ""
            section_title = path[-1] if path else ""
            prefix = f"文档章节：{' > '.join(path)}\n\n" if path else ""
            contextual = prefix + content
            chunks.append(DocumentChunk(
                index=len(chunks),
                content=content,
                contextual_content=contextual,
                section_title=section_title,
                section_path=path,
                section_index=section_index,
                token_count=estimate_tokens(content),
                content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            ))
            return _tail_within_budget(content, overlap)

        for piece in pieces:
            candidate = "\n\n".join([*current, piece])
            if current and estimate_tokens(candidate) > maximum:
                tail = emit()
                current = [tail] if tail else []
                if current and estimate_tokens("\n\n".join([*current, piece])) > maximum:
                    current = []
            current.append(piece)
        emit()
    return chunks


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """兼容旧调用方的字符窗口切分；新知识导入应使用 ``chunk_document``。"""
    text = " ".join(text.split())
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap。")
    return [text[start:start + chunk_size] for start in range(0, len(text), chunk_size - overlap)]
