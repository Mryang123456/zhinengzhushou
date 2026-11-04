"""多格式文档提取；依赖均为可选且在使用时加载。"""

from pathlib import Path


def extract_document(path: Path) -> str:
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
    raise ValueError("支持 txt、md、json、html、pdf、docx、xlsx。")
