"""知识库导入命令：python -m backend.app.knowledge.cli 文档路径。"""

import argparse
import os
from pathlib import Path
from backend.app.knowledge.ingestion import extract_document
from backend.app.knowledge.store import KnowledgeStore

parser = argparse.ArgumentParser(description="导入文本知识库")
parser.add_argument("path", type=Path)
parser.add_argument("--owner-id", required=True, help="文档所属用户或租户标识")
parser.add_argument("--scope", choices=("private", "shared"), default="private", help="private 仅所属用户可检索；shared 所有用户可检索")
args = parser.parse_args()
url = os.getenv("KNOWLEDGE_DATABASE_URL")
if not url:
    raise SystemExit("请先设置 KNOWLEDGE_DATABASE_URL。")
count = KnowledgeStore(url).ingest(str(args.path.resolve()), extract_document(args.path), args.owner_id, args.scope, {"filename": args.path.name, "suffix": args.path.suffix.lower()})
print(f"已导入 {count} 个文本块。")
