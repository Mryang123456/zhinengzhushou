"""最小可用的结构化日志、Trace 上下文和 Prometheus 风格指标。"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from contextvars import ContextVar

trace_id: ContextVar[str] = ContextVar("trace_id", default="-")
metrics: Counter[str] = Counter()
duration_sums: Counter[str] = Counter()
duration_counts: Counter[str] = Counter()
logger = logging.getLogger("langgraph_agent.events")


def increment(name: str, amount: int = 1) -> None:
    """增加低基数计数指标。"""
    if amount > 0:
        metrics[name] += amount


def event(name: str, **fields: object) -> None:
    increment(name)
    logger.info(json.dumps({"event": name, "trace_id": trace_id.get(), "at_ms": int(time.time() * 1000), **fields}, ensure_ascii=False))


def observe_duration(name: str, seconds: float) -> None:
    """记录低基数操作耗时；由 Prometheus 在实例级 scrape 后统一聚合。"""
    safe_seconds = max(0.0, seconds)
    duration_sums[name] += safe_seconds
    duration_counts[name] += 1


def prometheus_metrics() -> str:
    events = [f"agent_events_total{{event=\"{name}\"}} {count}" for name, count in sorted(metrics.items())]
    durations = [
        f"agent_operation_duration_seconds_sum{{operation=\"{name}\"}} {duration_sums[name]:.6f}\n"
        f"agent_operation_duration_seconds_count{{operation=\"{name}\"}} {duration_counts[name]}"
        for name in sorted(duration_counts)
    ]
    return "\n".join([*events, *durations]) + "\n"
