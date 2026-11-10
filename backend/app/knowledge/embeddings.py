"""Embedding 适配；默认不隐式调用付费模型。"""

import os


class EmbeddingUnavailable(RuntimeError):
    pass


def embed_texts(texts: list[str], *, for_search: bool = False) -> list[list[float]]:
    """仅在明确配置 openai 时调用 Embedding API。"""
    if os.getenv("KNOWLEDGE_EMBEDDING_PROVIDER", "").lower() != "openai":
        raise EmbeddingUnavailable("未启用 Embedding；将使用关键词检索。")
    from langchain_openai import OpenAIEmbeddings
    from openai import OpenAIError

    options = {"request_timeout": 5.0, "max_retries": 0} if for_search else {}
    try:
        model = OpenAIEmbeddings(model=os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "text-embedding-3-small"), **options)
        return model.embed_documents(texts)
    except (OpenAIError, TimeoutError, ConnectionError) as error:
        if not for_search:
            raise
        raise EmbeddingUnavailable("查询向量暂不可用，将使用关键词检索。") from error
