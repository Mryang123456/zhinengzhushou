"""向 Web 前端提供 Agent 的 SSE 流式接口。"""

import json
import logging
import os
import asyncio
import time
import uuid
from contextlib import aclosing, asynccontextmanager, nullcontext, suppress
from pathlib import Path

from anyio import CancelScope
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field, model_validator

from backend.app.agent.service import build_agent, continue_after_approved_mcp
from backend.app.agent.turn_gate import ThreadTurnGate, TurnQueueTimeout
from backend.app.agent.turns import checkpoint_config, checkpoint_scope, stream_events
from backend.app.api.chat_run_store import ChatRunStore, SubmissionConflict, TERMINAL
from backend.app.api.chat_run_worker import run_chat_worker
from backend.app.config import MCP_APPROVAL_DB_FILE, MCP_CONFIG_FILE, MEMORY_DB_FILE
from backend.app.mcp.approval import create_approval_store
from backend.app.mcp.governance import McpOutcomeUnknown, RISK_ORDER, invoke_governed, is_registered_tool, load_governance, load_mcp_connections, risk_for_tool, validate_tool_arguments
from backend.app.models.factory import ModelConfigurationError, ensure_model_ready, get_model_settings
from backend.app.security import AuthenticationError, AuthenticatedUser, authenticate_bearer, auth_required, create_rate_limiter, issue_access_token, login_rate_identity, validate_multi_instance_configuration, validate_production_configuration, valid_api_key, verify_bootstrap_credentials
from backend.app.observability import event, observe_duration, prometheus_metrics, trace_id
from backend.app.content_safety import is_direct_prompt_override


@asynccontextmanager
async def lifespan(application: FastAPI):
    """本地与 PostgreSQL Checkpoint 都由应用生命周期统一管理。"""
    database_url = os.getenv("CHECKPOINT_DATABASE_URL")
    application.state.checkpointer = MemorySaver() if not database_url else None
    application.state.checkpoint_pool = None
    application.state.chat_runs = None
    pool = None
    workers = []
    if database_url:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg_pool import AsyncConnectionPool
            pool_size = max(2, int(os.getenv("CHECKPOINT_POOL_MAX_SIZE", "10")))
            pool = AsyncConnectionPool(
                conninfo=database_url,
                kwargs={"autocommit": True, "prepare_threshold": 0},
                min_size=1,
                max_size=pool_size,
                open=False,
            )
            await pool.open()
            checkpointer = AsyncPostgresSaver(pool)
            async with pool.connection() as migration:
                await migration.execute("SELECT pg_advisory_lock(764814924190)")
                try:
                    await AsyncPostgresSaver(migration).setup()
                finally:
                    await migration.execute("SELECT pg_advisory_unlock(764814924190)")
            store = ChatRunStore(pool)
            await store.setup()
            application.state.checkpoint_pool = pool
            application.state.checkpointer = checkpointer
            application.state.chat_runs = store
            worker_count = min(pool_size - 1, max(1, int(os.getenv("AGENT_RUN_WORKERS", "2"))))
            workers = [asyncio.create_task(run_chat_worker(store)) for _ in range(worker_count)]
        except Exception:
            # 已创建的连接池也必须关闭，避免启动失败后遗留连接。
            pool = pool or getattr(application.state, "checkpoint_pool", None)
            if pool:
                await pool.close()
            raise
    try:
        yield
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        application.state.chat_runs = None
        application.state.checkpointer = None
        pool = getattr(application.state, "checkpoint_pool", None)
        if pool:
            await pool.close()


app = FastAPI(title="基础智能体流式服务", lifespan=lifespan)
thread_turn_gate = ThreadTurnGate()
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("AGENT_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Agent-Api-Key", "Authorization"],
)
logger = logging.getLogger("langgraph_agent.server")
validate_multi_instance_configuration()
validate_production_configuration()
rate_limiter = create_rate_limiter(
    limit=int(os.getenv("AGENT_RATE_LIMIT", "20")),
    window_seconds=int(os.getenv("AGENT_RATE_WINDOW_SECONDS", "60")),
)
login_rate_limiter = create_rate_limiter(
    limit=int(os.getenv("LOGIN_RATE_LIMIT", "5")),
    window_seconds=int(os.getenv("LOGIN_RATE_WINDOW_SECONDS", "900")),
)
approval_store = create_approval_store(MCP_APPROVAL_DB_FILE)


@app.middleware("http")
async def trace_request(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    token = trace_id.set(request_id)
    started = time.monotonic()
    try:
        event("http_request", method=request.method, path=request.url.path)
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response
    finally:
        observe_duration("http_request", time.monotonic() - started)
        trace_id.reset(token)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(default="web-demo", min_length=1, max_length=120)
    message_id: uuid.UUID | None = None
    attempt_id: uuid.UUID | None = None
    retry: bool = False
    recover_submission: bool = False
    use_mcp: bool = True
    use_memory: bool = True


class ApprovalConfirmation(BaseModel):
    confirmation_token: str | None = Field(default=None, min_length=20, max_length=200)
    run_id: uuid.UUID | None = None

    @model_validator(mode="after")
    def require_approval_proof(self):
        if self.confirmation_token is None and self.run_id is None:
            raise ValueError("请提供确认令牌或原任务 ID。")
        return self


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


def service_key_valid(api_key: str | None) -> bool:
    return valid_api_key(api_key, os.getenv("AGENT_API_KEY"))


def verify_request(request: Request, api_key: str | None) -> None:
    """执行可选 API 密钥验证与进程内限流。"""
    if not service_key_valid(api_key):
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
async def create_token(credentials: TokenRequest, http_request: Request) -> dict[str, str | int]:
    """使用已配置的初始账号换取短时 JWT；可由 OIDC 替代。"""
    if not auth_required():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="当前是本地开发模式，不需要登录令牌。")
    client_host = http_request.client.host if http_request.client else "unknown"
    if not login_rate_limiter.allow(login_rate_identity(client_host, credentials.username)):
        event("login_rate_limit_rejected", client=client_host)
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="登录尝试过于频繁，请稍后再试。")
    user = verify_bootstrap_credentials(credentials.username, credentials.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误。")
    return {"access_token": issue_access_token(user), "token_type": "bearer", "expires_in": max(5, min(int(os.getenv("JWT_EXPIRES_MINUTES", "30")), 480)) * 60}


@app.get("/auth/me")
async def current_user(authorization: str | None = Header(default=None)) -> dict[str, object]:
    user = authenticated_user(authorization)
    event("auth_me_read", user_id=user.user_id)
    return {"user_id": user.user_id, "roles": list(user.roles), "authentication_required": auth_required()}


@app.get("/connection/status")
async def connection_status(
    response: Response,
    x_agent_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    """检查聊天凭据，独立于聊天限流和模型调用。"""
    response.headers["Cache-Control"] = "no-store"
    if not service_key_valid(x_agent_api_key):
        return {"status": "invalid-api-key" if x_agent_api_key else "api-key-required"}
    try:
        authenticate_bearer(authorization)
    except AuthenticationError:
        return {"status": "session-expired" if authorization else "login-required"}
    return {"status": "connected"}


def _run_stream(store, user_id: str, run_id: uuid.UUID, after: int = 0) -> StreamingResponse:
    async def generate():
        cursor = after
        last_sent = time.monotonic()
        metadata = json.dumps({"type": "run", "run_id": str(run_id)})
        yield f"event: run\ndata: {metadata}\n\n"
        while True:
            rows = await store.events(user_id, run_id, cursor)
            if not rows:
                return
            run_status = rows[0][0]
            for _, event_id, payload in rows:
                if event_id is None:
                    continue
                cursor = event_id
                frame = json.dumps({**payload, "event_id": str(event_id)}, ensure_ascii=False)
                yield f"id: {event_id}\nevent: {payload['type']}\ndata: {frame}\n\n"
                last_sent = time.monotonic()
                if payload["type"] in {"done", "error"}:
                    return
            if rows[0][1] is not None:
                continue
            if run_status in TERMINAL:
                return
            if time.monotonic() - last_sent >= 15:
                yield ": keep-alive\n\n"
                last_sent = time.monotonic()
            await asyncio.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-store, no-transform", "X-Accel-Buffering": "no", "X-Content-Type-Options": "nosniff",
    })


@app.get("/chat/runs/{run_id}/events")
async def resume_chat_run(
    run_id: uuid.UUID, http_request: Request,
    after: int = Query(default=0, ge=0, le=9223372036854775807),
    x_agent_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    verify_request(http_request, x_agent_api_key)
    user = authenticated_user(authorization)
    store = getattr(app.state, "chat_runs", None)
    if store is None or await store.status(user.user_id, run_id) is None:
        raise HTTPException(404, detail="聊天任务不存在。")
    return _run_stream(store, user.user_id, run_id, after)


@app.post("/chat/runs/{run_id}/cancel")
async def cancel_chat_run(
    run_id: uuid.UUID, http_request: Request,
    x_agent_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    verify_request(http_request, x_agent_api_key)
    user = authenticated_user(authorization)
    store = getattr(app.state, "chat_runs", None)
    if store is None or await store.cancel(user.user_id, run_id) is None:
        raise HTTPException(404, detail="聊天任务不存在。")
    return {"accepted": True}


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
    if (request.retry or request.recover_submission) and request.message_id is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="重试必须提供原问题的消息 ID。")
    if is_direct_prompt_override(request.question):
        event("prompt_override_rejected", user_id=user.user_id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请求包含试图覆盖系统安全规则的内容，已拒绝处理。")
    try:
        ensure_model_ready(get_model_settings())
    except ModelConfigurationError as error:
        raise HTTPException(status_code=500, detail=f"模型配置错误：{error}") from error

    store = getattr(app.state, "chat_runs", None)
    if store is not None:
        if request.recover_submission and request.retry and request.attempt_id is None:
            raise HTTPException(422, detail="找回重试任务必须提供原尝试 ID。")
        if ":" in request.thread_id:
            raise HTTPException(422, detail="会话 ID 不能包含冒号。")
        message_id = request.message_id or uuid.uuid4()
        attempt_id = (request.attempt_id or uuid.uuid4()) if request.retry else message_id
        try:
            run_id = await store.submit(user.user_id, request.thread_id, message_id, attempt_id, {
                "question": request.question, "retry": request.retry,
                "use_memory": request.use_memory, "use_mcp": request.use_mcp,
            })
        except SubmissionConflict as error:
            raise HTTPException(409, detail=str(error)) from error
        return _run_stream(store, user.user_id, run_id)

    async def run_turn(deadline: asyncio.Timeout):
        thread_key = checkpoint_config(request.thread_id, user.user_id)["configurable"]["thread_id"]
        pool = getattr(app.state, "checkpoint_pool", None)
        wait_seconds = float(os.getenv("AGENT_THREAD_QUEUE_TIMEOUT_SECONDS", "120"))
        async with thread_turn_gate.hold(thread_key, wait_seconds, pool) as connection:
            run_seconds = float(os.getenv("AGENT_REQUEST_TIMEOUT_SECONDS", "90"))
            deadline.reschedule(asyncio.get_running_loop().time() + run_seconds)
            mcp_config = MCP_CONFIG_FILE if request.use_mcp else None
            memory_db = MEMORY_DB_FILE if request.use_memory else None
            shared_checkpointer = getattr(app.state, "checkpointer", None)
            if connection is not None:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                # 用持锁连接写 Checkpoint，连接断开时本轮也无法继续写入。
                shared_checkpointer = AsyncPostgresSaver(connection)
            checkpointer_context = (
                nullcontext(shared_checkpointer) if shared_checkpointer is not None
                else checkpoint_scope(os.getenv("CHECKPOINT_DATABASE_URL"))
            )
            async with checkpointer_context as checkpointer:
                graph = await build_agent(
                    mcp_config,
                    user.user_id,
                    memory_db,
                    os.getenv("MEMORY_DATABASE_URL") if request.use_memory else None,
                    checkpointer,
                    initial_question=request.question,
                )
                async with aclosing(stream_events(
                    graph, request.question, request.thread_id, user.user_id,
                    message_id=str(request.message_id) if request.message_id else None,
                    retry=request.retry or request.recover_submission,
                )) as frames:
                    async for frame in frames:
                        yield frame

    async def generate():
        try:
            yield 'event: progress\ndata: {"type":"progress","stage":"preparing"}\n\n'
            async with asyncio.timeout(None) as deadline:
                frames = run_turn(deadline)
                next_event = asyncio.create_task(anext(frames))
                try:
                    while True:
                        done, _ = await asyncio.wait({next_event}, timeout=15)
                        if await http_request.is_disconnected():
                            logger.info("客户端已断开，取消任务：thread_id=%s", request.thread_id)
                            return
                        if not done:
                            yield ": keep-alive\n\n"
                            continue
                        try:
                            frame = next_event.result()
                        except StopAsyncIteration:
                            break
                        yield f"event: {frame['type']}\ndata: {json.dumps(frame, ensure_ascii=False)}\n\n"
                        next_event = asyncio.create_task(anext(frames))
                finally:
                    with CancelScope(shield=True):
                        if not next_event.done():
                            next_event.cancel()
                        with suppress(asyncio.CancelledError, Exception):
                            await next_event
                        with suppress(Exception):
                            await frames.aclose()
        except TurnQueueTimeout:
            frame = {"type": "error", "message": "等待同一会话的上一轮完成超时，请稍后重试。", "retryable": True}
            yield f"event: error\ndata: {json.dumps(frame, ensure_ascii=False)}\n\n"
        except TimeoutError:
            event("agent_timeout", user_id=user.user_id, thread_id=request.thread_id)
            frame = {"type": "error", "message": "任务执行超时，已安全停止。"}
            yield f"event: error\ndata: {json.dumps(frame, ensure_ascii=False)}\n\n"
        except Exception as error:
            event("agent_failed", user_id=user.user_id, error=type(error).__name__)
            logger.exception("智能体流式请求失败：thread_id=%s user_id=%s", request.thread_id, user.user_id)
            frame = {"type": "error", "message": "智能体执行失败，请稍后重试或查看服务日志。"}
            yield f"event: error\ndata: {json.dumps(frame, ensure_ascii=False)}\n\n"

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
    token = confirmation.confirmation_token
    store = getattr(app.state, "chat_runs", None)
    if token is None and store is not None and confirmation.run_id is not None:
        token = await store.approval_token(user.user_id, confirmation.run_id, approval_id)
    action = approval_store.consume(approval_id, user.user_id, token) if token else None
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
    except McpOutcomeUnknown as error:
        logger.warning("无法确认 MCP 操作结果：approval_id=%s", approval_id)
        approval_store.complete(approval_id, str(error))
        raise HTTPException(status_code=502, detail=str(error)) from error
    except Exception:
        logger.exception("已确认的 MCP 操作执行失败：approval_id=%s", approval_id)
        detail = "已确认的工具操作执行失败或未返回可用结果。"
        approval_store.complete(approval_id, detail)
        raise HTTPException(status_code=502, detail=detail)
    try:
        final_answer = await continue_after_approved_mcp(action["question"], action["tool"], result)
    except Exception:
        logger.exception("已确认 MCP 操作的续答生成失败：approval_id=%s", approval_id)
        final_answer = "工具已执行，但无法生成经校验的说明；请稍后重新查询执行结果。"
    approval_store.complete(approval_id, final_answer)
    return {"status": "completed", "result": final_answer}
