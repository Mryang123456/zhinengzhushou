CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS agent_memories (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL CHECK (char_length(content) BETWEEN 1 AND 500),
    category TEXT NOT NULL DEFAULT 'fact',
    importance SMALLINT NOT NULL DEFAULT 3 CHECK (importance BETWEEN 1 AND 5),
    embedding vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ
);

ALTER TABLE agent_memories ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_agent_memories_user_active
    ON agent_memories (user_id, id DESC)
    WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_agent_memories_expiry
    ON agent_memories (expires_at)
    WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS knowledge_documents (
    id BIGSERIAL PRIMARY KEY, source TEXT NOT NULL, content_hash TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb, owner_id TEXT NOT NULL DEFAULT 'system',
    access_scope TEXT NOT NULL DEFAULT 'private' CHECK (access_scope IN ('private', 'shared')),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_documents_active_owner_source
    ON knowledge_documents(owner_id, source) WHERE status = 'active';
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id BIGSERIAL PRIMARY KEY, document_id BIGINT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL, content TEXT NOT NULL, embedding vector(1536), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE(document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding ON knowledge_chunks USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_knowledge_documents_access ON knowledge_documents(owner_id, access_scope) WHERE status = 'active';
