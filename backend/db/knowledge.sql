CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS knowledge_documents (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    owner_id TEXT NOT NULL DEFAULT 'system',
    access_scope TEXT NOT NULL DEFAULT 'private' CHECK (access_scope IN ('private', 'shared')),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT 'system';
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS access_scope TEXT NOT NULL DEFAULT 'private';
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';
ALTER TABLE knowledge_documents DROP CONSTRAINT IF EXISTS knowledge_documents_source_key;
CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_documents_active_owner_source
    ON knowledge_documents(owner_id, source) WHERE status = 'active';
DROP INDEX IF EXISTS idx_knowledge_documents_owner_source;
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding ON knowledge_chunks USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_content ON knowledge_chunks USING gin (to_tsvector('simple', content));
CREATE INDEX IF NOT EXISTS idx_knowledge_documents_access ON knowledge_documents(owner_id, access_scope) WHERE status = 'active';
