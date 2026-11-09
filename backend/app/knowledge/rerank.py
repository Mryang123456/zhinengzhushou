"""无额外模型时的可解释重排序；可替换为 Cross Encoder。"""

import re


def lexical_terms(text: str) -> set[str]:
    """中英文混合分词：中文加入二元词，避免整句中文只被当作一个词。"""
    normalized = text.lower()
    words = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    for run in chinese_runs:
        if len(run) == 1:
            words.add(run)
        else:
            words.update(run[index:index + 2] for index in range(len(run) - 1))
    return words


def reciprocal_rank_fusion(rankings: list[list[dict]], limit: int = 20, k: int = 60) -> list[dict]:
    """融合向量与关键词候选，避免任一召回通道独占结果。"""
    merged: dict[tuple, dict] = {}
    for ranking in rankings:
        for position, row in enumerate(ranking, start=1):
            key = (row.get("document_id"), row.get("chunk"))
            item = merged.setdefault(key, {**row, "score": 0.0})
            item["score"] += 1 / (k + position)
    return sorted(merged.values(), key=lambda row: row["score"], reverse=True)[:limit]

def rerank(query: str, rows: list[dict], limit: int = 5) -> list[dict]:
    terms = lexical_terms(query)
    for row in rows:
        overlap = len(terms & lexical_terms(row["content"]))
        row["score"] = round(float(row["score"]) + 0.08 * overlap, 4)
    return sorted(rows, key=lambda row: row["score"], reverse=True)[:limit]
