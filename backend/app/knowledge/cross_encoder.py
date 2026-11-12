"""本地 Cross-Encoder 片段及证据组重排。"""

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from threading import BoundedSemaphore

from backend.app.knowledge.rerank import unique_evidence_groups

logger = logging.getLogger(__name__)
CANDIDATE_LIMIT = 20
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="knowledge-rerank")
_slot = BoundedSemaphore(1)


@lru_cache(maxsize=1)
def _load_model(model_path: str, device: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        model_path, device=device, max_length=2048, local_files_only=True,
        trust_remote_code=False, model_kwargs={"use_safetensors": True},
    )


def _rank(model, query: str, rows: list[dict], passages: list[str], score_key: str) -> list[dict]:
    if len(rows) < 2:
        return rows
    scores = [
        float(score) for score in model.predict(
            [(query, passage) for passage in passages], batch_size=4, show_progress_bar=False,
        )
    ]
    if len(scores) != len(rows) or not all(math.isfinite(score) for score in scores):
        raise ValueError("重排模型返回了无效分数。")
    ranked = [{**row, score_key: score} for row, score in zip(rows, scores, strict=True)]
    return sorted(ranked, key=lambda row: row[score_key], reverse=True)


def _group_text(row: dict) -> str:
    heading = " > ".join(row.get("section_path") or ())
    return "\n\n".join([heading, *(item["content"] for item in row["evidence_chunks"])])


def _rank_both(model_path: str, device: str, query: str, groups: list[dict], limit: int):
    model = _load_model(model_path, device)
    candidates = groups[:CANDIDATE_LIMIT]
    query = query[:1000]
    seeds = _rank(
        model, query, candidates,
        [row.get("contextual_content") or row["content"] for row in candidates],
        "cross_encoder_score",
    )
    shortlist = unique_evidence_groups(seeds)[:min(limit * 2, CANDIDATE_LIMIT)]
    ranked_groups = _rank(
        model, query, shortlist, [_group_text(row) for row in shortlist],
        "evidence_group_score",
    )
    tail = groups[CANDIDATE_LIMIT:]
    return (seeds + tail)[:limit], unique_evidence_groups(ranked_groups + tail)[:limit]


def rerank_evidence(model_path: str, query: str, groups: list[dict], limit: int):
    """限制每进程最多一个推理任务，超时后让当前推理结束，避免请求积压。"""
    if len(groups) < 2:
        return None, "skipped"
    try:
        timeout = float(os.getenv("KNOWLEDGE_RERANK_TIMEOUT_SECONDS", "5"))
        if not math.isfinite(timeout) or not 0.1 <= timeout <= 30:
            raise ValueError("重排等待时间必须在 0.1 到 30 秒之间。")
        if not _slot.acquire(blocking=False):
            return None, "fallback"
        try:
            future = _executor.submit(
                _rank_both, model_path, os.getenv("KNOWLEDGE_RERANK_DEVICE", "cpu"),
                query, groups, limit,
            )
        except BaseException:
            _slot.release()
            raise
        future.add_done_callback(lambda _: _slot.release())
        return future.result(timeout=timeout), "cross_encoder"
    except Exception as error:
        logger.warning("知识库重排回退到规则排序：%s", type(error).__name__)
        return None, "fallback"
