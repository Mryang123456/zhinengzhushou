"""向 Web 前端提供 Agent 的 SSE 流式接口。"""

import json
import logging
import os
import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app.agent.service import build_agent, checkpoint_scope, continue_after_approved_mcp, load_mcp_connections, stream_events
from backend.app.config import MCP_APPROVAL_DB_FILE, MCP_CONFIG_FILE, MEMORY_DB_FILE
from backend.app.mcp.approval import create_approval_store
from backend.app.mcp.governance import RISK_ORDER, invoke_governed, is_registered_tool, load_governance, risk_for_tool, validate_tool_arguments
from backend.app.models.factory import ModelConfigurationError, ensure_model_ready, get_model_settings
from backend.app.security import AuthenticationError, AuthenticatedUser, authenticate_bearer, auth_required, create_rate_limiter, issue_access_token, validate_multi_instance_configuration, valid_api_key, verify_bootstrap_credentials
from backend.app.observability import event, prometheus_metrics, trace_id


app = FastAPI(title="基础智能体流式服务")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("AGENT_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Agent-Api-Key", "Authorization"],
)
logger = logging.getLogger("langgraph_agent.server")
validate_multi_instance_configuration()
rate_limiter = create_rate_limiter(
    limit=int(os.getenv("AGENT_RATE_LIMIT", "20")),
    window_seconds=int(os.getenv("AGENT_RATE_WINDOW_SECONDS", "60")),
)
approval_store = create_approval_store(MCP_APPROVAL_DB_FILE)


@app.middleware("http")
async def trace_request(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    token = trace_id.set(request_id)
    try:
        event("http_request", method=request.method, path=request.url.path)
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response
    finally:
        trace_id.reset(token)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(default="web-demo", min_length=1, max_length=120)
    use_mcp: bool = True
    use_memory: bool = True


class ApprovalConfirmation(BaseModel):
    confirmation_token: str = Field(min_length=20, max_length=200)


class TokenRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=256)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """仅检查服务进程；不触发模型、MCP 或外部网络调用。"""
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
async def metrics_endpoint():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(prometheus_metrics(), media_type="text/plain; version=0.0.4")


def verify_request(request: Request, api_key: str | None) -> None:
    """执行可选 API 密钥验证与进程内限流。"""
    expected_key = os.getenv("AGENT_API_KEY")
    if not valid_api_key(api_key, expected_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="访问密钥无效。")
    client_host = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_host):
        event("rate_limit_rejected", client=client_host)
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="请求过于频繁，请稍后重试。")


def authenticated_user(authorization: str | None) -> AuthenticatedUser:
    try:
        return authenticate_bearer(authorization)
    except AuthenticationError as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(error)) from error


@app.post("/auth/token")
async def create_token(credentials: TokenRequest) -> dict[str, str | int]:
    """使用已配置的初始账号换取短时 JWT；可由 OIDC 替代。"""
    if not auth_required():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="当前是本地开发模式，不需要登录令牌。")
    user = verify_bootstrap_credentials(credentials.username, credentials.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误。")
    return {"access_token": issue_access_token(user), "token_type": "bearer", "expires_in": max(5, min(int(os.getenv("JWT_EXPIRES_MINUTES", "30")), 480)) * 60}


@app.get("/auth/me")
async def current_user(authorization: str | None = Header(default=None)) -> dict[str, object]:
    user = authenticated_user(authorization)
    event("agent_started", user_id=user.user_id, thread_id=request.thread_id)
    return {"user_id": user.user_id, "roles": list(user.roles), "authentication_required": auth_required()}


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
    x_agent_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> StreamingResponse:
    """以 SSE 依次发送 token、工具事件、完成事件或错误事件。"""
    verify_request(http_request, x_agent_api_key)
    user = authenticated_user(authorization)
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
                    user.user_id,
                    memory_db,
                    os.getenv("MEMORY_DATABASE_URL") if request.use_memory else None,
                    checkpointer,
                )
                async with asyncio.timeout(float(os.getenv("AGENT_REQUEST_TIMEOUT_SECONDS", "90"))):
                    async for event in stream_events(
                        graph, request.question, request.thread_id, user.user_id
                    ):
                        if await http_request.is_disconnected():
                            logger.info("客户端已断开，取消任务：thread_id=%s", request.thread_id)
                            return
                        yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        except TimeoutError:
            event("agent_timeout", user_id=user.user_id, thread_id=request.thread_id)
            event = {"type": "error", "message": "任务执行超时，已安全停止。"}
            yield f"event: error\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as error:
            event("agent_failed", user_id=user.user_id, error=type(error).__name__)
            logger.exception("智能体流式请求失败：thread_id=%s user_id=%s", request.thread_id, user.user_id)
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
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    """消费一次性确认令牌后，执行之前被拦截的高风险 MCP 工具。"""
    verify_request(http_request, x_agent_api_key)
    user = authenticated_user(authorization)
    action = approval_store.consume(approval_id, user.user_id, confirmation.confirmation_token)
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
            user.user_id,
            f"approval:{approval_id}",
            governance.get("timeoutSeconds", 15),
            governance.get("retries", 1),
            approved=True,
            approval_id=approval_id,
        )
    except Exception:
        logger.exception("已确认的 MCP 操作执行失败：approval_id=%s", approval_id)
        result = "已确认的工具操作执行失败，未返回可用结果。"
    try:
        final_answer = await continue_after_approved_mcp(action["question"], action["tool"], result)
    except Exception:
        logger.exception("已确认 MCP 操作的续答生成失败：approval_id=%s", approval_id)
        final_answer = "工具已执行，但无法生成经校验的说明；请稍后重新查询执行结果。"
    approval_store.complete(approval_id, final_answer)
    return {"status": "completed", "result": final_answer}
