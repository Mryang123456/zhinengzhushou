"""PostgreSQL + pgvector 知识库实现，含关键词降级检索。"""

import hashlib
import json
from pathlib import Path

from backend.app.config import KNOWLEDGE_SCHEMA_FILE
from backend.app.knowledge.chunking import split_text
from backend.app.knowledge.embeddings import EmbeddingUnavailable, embed_texts
from backend.app.knowledge.rerank import rerank


class KnowledgeStore:
    def __init__(self, database_url: str):
        import psycopg
        self.database_url, self.psycopg = database_url, psycopg
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(KNOWLEDGE_SCHEMA_FILE.read_text(encoding="utf-8"))

    def _connect(self):
        return self.psycopg.connect(self.database_url)

    def ingest(self, source: str, text: str, metadata: dict | None = None) -> int:
        chunks = split_text(text)
        if not chunks:
            raise ValueError("文档没有可导入文本。")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            vectors = embed_texts(chunks)
        except EmbeddingUnavailable:
            vectors = [None] * len(chunks)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("INSERT INTO knowledge_documents(source, content_hash, metadata) VALUES (%s, %s, %s) ON CONFLICT (source) DO UPDATE SET content_hash = EXCLUDED.content_hash, metadata = EXCLUDED.metadata, updated_at = NOW() RETURNING id", (source, digest, json.dumps(metadata or {})))
            document_id = cursor.fetchone()[0]
            cursor.execute("DELETE FROM knowledge_chunks WHERE document_id = %s", (document_id,))
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                vector_text = "[" + ",".join(map(str, vector)) + "]" if vector else None
                cursor.execute("INSERT INTO knowledge_chunks(document_id, chunk_index, content, embedding) VALUES (%s, %s, %s, %s::vector)", (document_id, index, chunk, vector_text))
        return len(chunks)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        try:
            vector = embed_texts([query])[0]
        except EmbeddingUnavailable:
            vector = None
        with self._connect() as connection, connection.cursor() as cursor:
            if vector:
                vector_text = "[" + ",".join(map(str, vector)) + "]"
                cursor.execute("SELECT d.source, c.chunk_index, c.content, 1 - (c.embedding <=> %s::vector) AS score FROM knowledge_chunks c JOIN knowledge_documents d ON d.id=c.document_id WHERE c.embedding IS NOT NULL ORDER BY c.embedding <=> %s::vector LIMIT %s", (vector_text, vector_text, limit))
            else:
                cursor.execute("SELECT d.source, c.chunk_index, c.content, CASE WHEN c.content ILIKE %s THEN 1.0 ELSE 0.0 END AS score FROM knowledge_chunks c JOIN knowledge_documents d ON d.id=c.document_id ORDER BY score DESC, c.id DESC LIMIT %s", (f"%{query.strip()}%", limit))
            rows = [{"source": row[0], "chunk": row[1], "content": row[2], "score": float(row[3])} for row in cursor.fetchall()]
        return rerank(query, rows, limit)


def read_text_file(path: Path) -> str:
    if path.suffix.lower() not in {".txt", ".md", ".json"}:
        raise ValueError("基础版仅支持 .txt、.md、.json 文档。")
    return path.read_text(encoding="utf-8")
