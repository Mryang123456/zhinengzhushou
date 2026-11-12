"""保留文档结构的多格式文本提取；依赖在使用时加载。"""

from __future__ import annotations

import json
import os
from pathlib import Path


SUPPORTED_SUFFIXES = {".txt", ".md", ".json", ".html", ".htm", ".pdf", ".docx", ".xlsx"}


def validate_document_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError("知识库导入目标必须是可读取的文件。")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("支持 txt、md、json、html、pdf、docx、xlsx。")
    maximum = max(1, int(os.getenv("KNOWLEDGE_MAX_DOCUMENT_BYTES", str(10 * 1024 * 1024))))
    if path.stat().st_size > maximum:
        raise ValueError(f"文档超过大小限制（{maximum} 字节）。")


def extract_document(path: Path) -> str:
    """提取为带 Markdown 标题的纯文本，使下游切片器能够识别章节边界。"""
    validate_document_file(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".json":
        raw = path.read_text(encoding="utf-8")
        try:
            return json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
        except json.JSONDecodeError as error:
            raise ValueError("JSON 文档格式无效。") from error
    if suffix in {".html", ".htm"}:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                heading.string = f"{'#' * level} {heading.get_text(' ', strip=True)}"
        return soup.get_text("\n", strip=True)
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(
            f"# 第 {index} 页\n\n{page.extract_text() or ''}"
            for index, page in enumerate(PdfReader(path).pages, start=1)
        )
    if suffix == ".docx":
        from docx import Document
        document = Document(path)
        parts: list[str] = []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            style = (paragraph.style.name or "").lower()
            if style.startswith("heading"):
                digits = "".join(character for character in style if character.isdigit())
                level = min(max(int(digits or "1"), 1), 6)
                parts.append(f"{'#' * level} {text}")
            else:
                parts.append(text)
        for table_index, table in enumerate(document.tables, start=1):
            parts.append(f"## 表格 {table_index}")
            parts.extend(" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows)
        return "\n\n".join(parts)
    if suffix == ".xlsx":
        from openpyxl import load_workbook
        book = load_workbook(path, read_only=True, data_only=True)
        parts: list[str] = []
        try:
            for sheet in book.worksheets:
                parts.append(f"# 工作表：{sheet.title}")
                for row in sheet.iter_rows(values_only=True):
                    values = [str(value).strip() if value is not None else "" for value in row]
                    if any(values):
                        parts.append(" | ".join(values))
        finally:
            book.close()
        return "\n".join(parts)
    raise ValueError("不支持的文档格式。")
