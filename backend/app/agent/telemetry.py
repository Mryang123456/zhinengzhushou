"""记录工作流节点耗时，并把耗时事件交给流式输出。"""

from __future__ import annotations

from contextlib import asynccontextmanager
import time
from typing import AsyncIterator

from langgraph.config import get_stream_writer

from backend.app.observability import observe_duration


@asynccontextmanager
async def observe_async_operation(name: str) -> AsyncIterator[None]:
    started = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - started
        observe_duration(name, elapsed)
        get_stream_writer()({
            "type": "timing",
            "operation": name,
            "elapsed_ms": round(elapsed * 1000),
        })
