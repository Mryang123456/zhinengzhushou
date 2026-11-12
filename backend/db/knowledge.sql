CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS knowledge_documents (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    owner_id TEXT NOT NULL DEFAULT 'system',
    access_scope TEXT NOT NULL DEFAULT 'private' CHECK (access_scope IN ('private', 'shared')),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')),
    chunking_version TEXT NOT NULL DEFAULT 'legacy-v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT 'system';
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS access_scope TEXT NOT NULL DEFAULT 'private';
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS chunking_version TEXT NOT NULL DEFAULT 'legacy-v1';
ALTER TABLE knowledge_documents DROP CONSTRAINT IF EXISTS knowledge_documents_source_key;
CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_documents_active_owner_source
    ON knowledge_documents(owner_id, source) WHERE status = 'active';
DROP INDEX IF EXISTS idx_knowledge_documents_owner_source;
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    contextual_content TEXT NOT NULL,
    section_title TEXT NOT NULL DEFAULT '',
    section_path TEXT[] NOT NULL DEFAULT '{}'::TEXT[],
    section_index INTEGER,
    token_count INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL DEFAULT '',
    embedding vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);
ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS contextual_content TEXT;
ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS section_title TEXT NOT NULL DEFAULT '';
ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS section_path TEXT[] NOT NULL DEFAULT '{}'::TEXT[];
ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS section_index INTEGER;
ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS token_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT '';
UPDATE knowledge_chunks SET contextual_content = content WHERE contextual_content IS NULL;
ALTER TABLE knowledge_chunks ALTER COLUMN contextual_content SET NOT NULL;
ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS search_vector tsvector
    GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(section_title, '') || ' ' || contextual_content)
    ) STORED;
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding ON knowledge_chunks USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL;
DROP INDEX IF EXISTS idx_knowledge_chunks_content;
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_search ON knowledge_chunks USING gin (search_vector);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_section_order
    ON knowledge_chunks(document_id, section_path, chunk_index);
CREATE INDEX IF NOT EXISTS idx_knowledge_documents_access ON knowledge_documents(owner_id, access_scope) WHERE status = 'active';
