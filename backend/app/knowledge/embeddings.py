"""Embedding 适配；默认不隐式调用付费模型。"""

import os


class EmbeddingUnavailable(RuntimeError):
    pass


def embed_texts(texts: list[str]) -> list[list[float]]:
    """仅在明确配置 openai 时调用 Embedding API。"""
    if os.getenv("KNOWLEDGE_EMBEDDING_PROVIDER", "").lower() != "openai":
        raise EmbeddingUnavailable("未启用 Embedding；将使用关键词检索。")
    from langchain_openai import OpenAIEmbeddings
    model = OpenAIEmbeddings(model=os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "text-embedding-3-small"))
    return model.embed_documents(texts)
