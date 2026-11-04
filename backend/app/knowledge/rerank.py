"""无额外模型时的可解释重排序；可替换为 Cross Encoder。"""

import re

def rerank(query: str, rows: list[dict], limit: int = 5) -> list[dict]:
    terms = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
    for row in rows:
        overlap = len(terms & set(re.findall(r"[\w\u4e00-\u9fff]+", row["content"].lower())))
        row["score"] = round(float(row["score"]) + 0.08 * overlap, 4)
    return sorted(rows, key=lambda row: row["score"], reverse=True)[:limit]
