"""无额外模型时的可解释重排序；可替换为 Cross Encoder。"""

import re


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
    terms = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
    for row in rows:
        overlap = len(terms & set(re.findall(r"[\w\u4e00-\u9fff]+", row["content"].lower())))
        row["score"] = round(float(row["score"]) + 0.08 * overlap, 4)
    return sorted(rows, key=lambda row: row["score"], reverse=True)[:limit]
