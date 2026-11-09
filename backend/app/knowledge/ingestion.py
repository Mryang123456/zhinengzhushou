"""多格式文档提取；依赖均为可选且在使用时加载。"""

from pathlib import Path
import os


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
    validate_document_file(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".json", ".html", ".htm"}:
        text = path.read_text(encoding="utf-8")
        if suffix in {".html", ".htm"}:
            from bs4 import BeautifulSoup
            return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
        return text
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        from docx import Document
        return "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    if suffix == ".xlsx":
        from openpyxl import load_workbook
        book = load_workbook(path, read_only=True, data_only=True)
        return "\n".join(" ".join(str(value) for row in sheet.iter_rows(values_only=True) for value in row if value is not None) for sheet in book.worksheets)
    raise ValueError("不支持的文档格式。")
