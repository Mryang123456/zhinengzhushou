"""知识库导入命令：python -m backend.app.knowledge.cli 文档路径。"""

import argparse
import os
from pathlib import Path
from backend.app.knowledge.ingestion import extract_document
from backend.app.knowledge.store import KnowledgeStore

parser = argparse.ArgumentParser(description="导入文本知识库")
parser.add_argument("path", type=Path)
args = parser.parse_args()
url = os.getenv("KNOWLEDGE_DATABASE_URL")
if not url:
    raise SystemExit("请先设置 KNOWLEDGE_DATABASE_URL。")
count = KnowledgeStore(url).ingest(str(args.path.resolve()), extract_document(args.path), {"filename": args.path.name, "suffix": args.path.suffix.lower()})
print(f"已导入 {count} 个文本块。")
