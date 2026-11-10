"""检索评测：比较召回候选、重排结果和最终交给模型的证据。"""

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import mean

from backend.app.knowledge.rerank import rank_candidates
from backend.app.knowledge.store import KnowledgeStore, knowledge_min_similarity


def recall(rows: list[dict], relevant: set[str]) -> float:
    return len(relevant & {row["citation"] for row in rows}) / len(relevant)


def measure(stages, relevant: set[str], k: int) -> dict[str, float]:
    if not relevant:
        return {"no_evidence_accuracy": float(not stages.results)}

    seen: set[str] = set()
    hits = []
    for row in stages.results[:k]:
        citation = row["citation"]
        hits.append(citation in relevant and citation not in seen)
        seen.add(citation)
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return {
        "candidate_recall": recall(stages.candidates, relevant),
        "reranked_recall_at_k": recall(stages.reranked[:k], relevant),
        "recall_at_k": sum(hits) / len(relevant),
        "precision_at_k": sum(hits) / k,
        "mrr_at_k": next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0),
        "ndcg_at_k": dcg / ideal,
    }


def load_cases(raw: bytes) -> list[dict]:
    cases = []
    seen = set()
    for line_no, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict):
            raise ValueError(f"第 {line_no} 行应为 JSON 对象。")
        case_id = case.get("id")
        question = case.get("question")
        relevant = case.get("relevant_chunks")
        if (
            not isinstance(case_id, str) or not case_id.strip() or case_id in seen
            or not isinstance(question, str) or not question.strip()
            or not isinstance(relevant, list)
            or any(not isinstance(citation, str) or not citation.strip() for citation in relevant)
        ):
            raise ValueError(f"第 {line_no} 行需要唯一 id、question 和 relevant_chunks。")
        seen.add(case_id)
        cases.append(case)
    if not cases:
        raise ValueError("评测集为空。")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--mode", choices=("fixture", "live"), required=True)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--min-recall", type=float)
    parser.add_argument("--min-no-evidence-accuracy", type=float)
    parser.add_argument("--require-hybrid", action="store_true")
    args = parser.parse_args()
    gates = {
        key: value for key, value in {
            "recall_at_k": args.min_recall,
            "no_evidence_accuracy": args.min_no_evidence_accuracy,
        }.items() if value is not None
    }
    if args.k < 1 or any(not 0 <= value <= 1 for value in gates.values()):
        parser.error("K 必须为正数，评测门槛必须在 [0, 1] 之间。")

    store = None
    if args.mode == "live":
        from dotenv import load_dotenv
        from backend.app.config import ENV_FILE

        load_dotenv(ENV_FILE)
        database_url = os.getenv("KNOWLEDGE_DATABASE_URL")
        if not database_url:
            parser.error("live 模式需要配置 KNOWLEDGE_DATABASE_URL。")
        store = KnowledgeStore(database_url, initialize_schema=False)

    minimum_similarity = knowledge_min_similarity() if store else 0.22
    raw = args.dataset.read_bytes()
    cases = load_cases(raw)
    metrics: dict[str, list[float]] = {}
    retrieval_modes: Counter[str] = Counter()
    details = []
    for case in cases:
        if store is not None:
            user_id = case.get("user_id")
            if not isinstance(user_id, str) or not user_id.strip():
                parser.error(f"live 模式的样本 {case['id']} 缺少 user_id。")
            stages = store.search_stages(case["question"], user_id, args.k, minimum_similarity)
        else:
            stages = rank_candidates(
                case["question"], case["vector_rows"], case["keyword_rows"],
                args.k, minimum_similarity,
            )
        retrieval_modes[stages.mode] += 1
        scores = measure(stages, set(case["relevant_chunks"]), args.k)
        for metric, value in scores.items():
            metrics.setdefault(metric, []).append(value)
        details.append({
            "id": case["id"], "retrieval_mode": stages.mode, **scores,
            "returned_chunks": [row["citation"] for row in stages.results],
        })
    averages = {metric: mean(values) for metric, values in metrics.items()}
    failures = [key for key, value in gates.items() if averages.get(key, -1) < value]
    if args.require_hybrid and retrieval_modes["hybrid"] != len(cases):
        failures.append("require_hybrid")
    print(json.dumps({
        "mode": args.mode,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "k": args.k,
        "minimum_similarity": minimum_similarity,
        "samples": len(cases),
        "answerable_samples": len(metrics.get("recall_at_k", [])),
        "retrieval_modes": retrieval_modes,
        "metrics": averages,
        "failed_gates": failures,
        "cases": details,
    }, ensure_ascii=False, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
