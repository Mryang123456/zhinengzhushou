"""向 Web 前端提供 Agent 的 SSE 流式接口。"""

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app.agent.service import build_agent, checkpoint_scope, load_mcp_connections, stream_events
from backend.app.config import MCP_APPROVAL_DB_FILE, MCP_CONFIG_FILE, MEMORY_DB_FILE
from backend.app.mcp.approval import ApprovalStore
from backend.app.mcp.governance import RISK_ORDER, invoke_governed, is_registered_tool, load_governance, risk_for_tool, validate_tool_arguments
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
approval_store = ApprovalStore(MCP_APPROVAL_DB_FILE)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(default="web-demo", min_length=1, max_length=120)
    user_id: str = Field(default="default", min_length=1, max_length=120)
    use_mcp: bool = True
    use_memory: bool = True


class ApprovalConfirmation(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    confirmation_token: str = Field(min_length=20, max_length=200)


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


@app.post("/mcp/approvals/{approval_id}/confirm")
async def confirm_mcp_approval(
    approval_id: str,
    confirmation: ApprovalConfirmation,
    http_request: Request,
    x_agent_api_key: str | None = Header(default=None),
) -> dict[str, str]:
    """消费一次性确认令牌后，执行之前被拦截的高风险 MCP 工具。"""
    verify_request(http_request, x_agent_api_key)
    action = approval_store.consume(approval_id, confirmation.user_id, confirmation.confirmation_token)
    if not action:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="该确认请求无效、已过期或已被使用。")

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        governance = load_governance(MCP_CONFIG_FILE)
        client = MultiServerMCPClient(load_mcp_connections(MCP_CONFIG_FILE), tool_name_prefix=True)
        tools = {tool.name: tool for tool in await client.get_tools()}
        tool = tools.get(action["tool"])
        if not tool or not is_registered_tool(tool.name, governance) or RISK_ORDER.get(risk_for_tool(tool.name, governance), 4) < RISK_ORDER["L2"]:
            raise RuntimeError("待执行的 MCP 工具不在受控高风险注册表中。")
        arguments = validate_tool_arguments(tool.name, action["arguments"], governance)
        result = await invoke_governed(
            tool,
            arguments,
            action["risk"],
            confirmation.user_id,
            f"approval:{approval_id}",
            governance.get("timeoutSeconds", 15),
            governance.get("retries", 1),
            approved=True,
            approval_id=approval_id,
        )
    except Exception:
        logger.exception("已确认的 MCP 操作执行失败：approval_id=%s", approval_id)
        result = "已确认的工具操作执行失败，未返回可用结果。"
    approval_store.complete(approval_id, result)
    return {"status": "completed", "result": result}
