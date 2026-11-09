"""最小可用的结构化日志、Trace 上下文和 Prometheus 风格指标。"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from contextvars import ContextVar

trace_id: ContextVar[str] = ContextVar("trace_id", default="-")
metrics: Counter[str] = Counter()
logger = logging.getLogger("langgraph_agent.events")


def event(name: str, **fields: object) -> None:
    metrics[name] += 1
    logger.info(json.dumps({"event": name, "trace_id": trace_id.get(), "at_ms": int(time.time() * 1000), **fields}, ensure_ascii=False))


def prometheus_metrics() -> str:
    return "\n".join(f"agent_events_total{{event=\"{name}\"}} {count}" for name, count in sorted(metrics.items())) + "\n"
