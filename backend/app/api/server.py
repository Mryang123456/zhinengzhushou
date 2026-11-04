"""向 Web 前端提供 Agent 的 SSE 流式接口。"""

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app.agent.service import build_agent, checkpoint_scope, stream_events
from backend.app.config import MCP_CONFIG_FILE, MEMORY_DB_FILE
from backend.app.models.factory import ModelConfigurationError, ensure_model_ready, get_model_settings
from backend.app.security import SlidingWindowRateLimiter, valid_api_key


app = FastAPI(title="基础智能体流式服务")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("AGENT_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Agent-Api-Key"],
)
logger = logging.getLogger("langgraph_agent.server")
rate_limiter = SlidingWindowRateLimiter(
    limit=int(os.getenv("AGENT_RATE_LIMIT", "20")),
    window_seconds=int(os.getenv("AGENT_RATE_WINDOW_SECONDS", "60")),
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(default="web-demo", min_length=1, max_length=120)
    user_id: str = Field(default="default", min_length=1, max_length=120)
    use_mcp: bool = True
    use_memory: bool = True


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """仅检查服务进程；不触发模型、MCP 或外部网络调用。"""
    return {"status": "ok"}


def verify_request(request: Request, api_key: str | None) -> None:
    """执行可选 API 密钥验证与进程内限流。"""
    expected_key = os.getenv("AGENT_API_KEY")
    if not valid_api_key(api_key, expected_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="访问密钥无效。")
    client_host = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_host):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="请求过于频繁，请稍后重试。")


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
    x_agent_api_key: str | None = Header(default=None),
) -> StreamingResponse:
    """以 SSE 依次发送 token、工具事件、完成事件或错误事件。"""
    verify_request(http_request, x_agent_api_key)
    try:
        ensure_model_ready(get_model_settings())
    except ModelConfigurationError as error:
        raise HTTPException(status_code=500, detail=f"模型配置错误：{error}") from error

    async def generate():
        try:
            mcp_config = MCP_CONFIG_FILE if request.use_mcp else None
            memory_db = MEMORY_DB_FILE if request.use_memory else None
            async with checkpoint_scope(os.getenv("CHECKPOINT_DATABASE_URL")) as checkpointer:
                graph = await build_agent(
                    mcp_config,
                    request.user_id,
                    memory_db,
                    os.getenv("MEMORY_DATABASE_URL") if request.use_memory else None,
                    checkpointer,
                )
                async for event in stream_events(
                    graph, request.question, request.thread_id, request.user_id
                ):
                    yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as error:
            logger.exception("智能体流式请求失败：thread_id=%s user_id=%s", request.thread_id, request.user_id)
            event = {"type": "error", "message": "智能体执行失败，请稍后重试或查看服务日志。"}
            yield f"event: error\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )
