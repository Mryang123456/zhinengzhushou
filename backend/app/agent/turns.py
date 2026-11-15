"""管理对话检查点、请求重试与流式回合。"""

from contextlib import aclosing, asynccontextmanager

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from backend.app.agent.plan_progress import plan_progress
from backend.app.observability import event
from backend.app.quality.web_activity import search_highlights
from backend.app.quality.validator import cited_web_sources, web_result_urls


def _done_event(answer: str, messages: list[BaseMessage]) -> dict:
    turn_start = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), 0)
    sources = cited_web_sources(answer, messages[turn_start:])
    return {"type": "done", "content": answer, **({"sources": sources} if sources else {})}


@asynccontextmanager
async def checkpoint_scope(database_url: str | None = None):
    """提供检查点存储；配置 PostgreSQL 时，任务状态可跨进程恢复。"""
    if not database_url:
        yield MemorySaver()
        return

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool
    except ImportError as error:
        raise RuntimeError(
            "缺少 PostgreSQL 检查点依赖，请重新安装 requirements.txt。"
        ) from error

    async with AsyncConnectionPool(
        conninfo=database_url,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        yield checkpointer


def checkpoint_config(thread_id: str, user_id: str = "default") -> dict:
    """生成按用户隔离的检查点键，防止不同用户共享同一会话状态。"""
    return {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}


async def ask(app, question: str, thread_id: str, user_id: str = "default") -> str:
    """执行一个回合；重用 thread_id 可在当前进程内保持对话。"""
    result = await app.ainvoke(
        {"messages": [HumanMessage(question)], "retry": False, "durable_run_id": None},
        config=checkpoint_config(thread_id, user_id),
    )
    return result["messages"][-1].content


async def stream_events(
    app, question: str, thread_id: str, user_id: str = "default", *,
    message_id: str | None = None, retry: bool = False,
    run_id: str | None = None,
):
    """逐步产出 token、工具状态和最终答案，供命令行或 SSE 接口复用。"""
    config = checkpoint_config(thread_id, user_id)
    stable_id = f"web:{message_id}" if message_id is not None else None
    question_message = HumanMessage(content=question, id=stable_id)
    messages: list[BaseMessage] = [question_message]
    seen_web_urls: set[str] = set()
    track_plan = False
    last_plan_frame = None
    input_state: dict | None = {"messages": messages, "retry": retry, "durable_run_id": run_id}
    if stable_id is not None:
        snapshot = await app.aget_state(config)
        history = snapshot.values.get("messages", [])
        if run_id and snapshot.values.get("durable_run_id") == run_id:
            if "intent" not in snapshot.next:
                track_plan = bool(snapshot.values.get("intent", {}).get("tasks"))
                if track_plan and (last_plan_frame := plan_progress(snapshot.values)):
                    yield last_plan_frame
            if snapshot.values.get("finished_run_id") == run_id and history and isinstance(history[-1], AIMessage):
                yield _done_event(history[-1].content, history)
                return
            input_state = None
            event("chat_checkpoint_resumed", compound=track_plan)
            start = next((i for i in range(len(history) - 1, -1, -1) if isinstance(history[i], HumanMessage)), 0)
            seen_web_urls = web_result_urls(history[start:]) or set()
        else:
            rejection = _prepare_retry(history, messages, stable_id, question, retry)
            if rejection:
                yield rejection
                return
    final_answer = ""
    returned_tools = []
    options = {"durability": "sync"} if run_id else {}
    async with aclosing(app.astream(
        input_state,
        config=config,
        stream_mode=["custom", "updates", "values"],
        **options,
    )) as events:
        async for mode, payload in events:
            if mode == "values":
                if track_plan:
                    frame = plan_progress(payload)
                    if frame and frame != last_plan_frame:
                        last_plan_frame = frame
                        yield frame
                continue
            if mode == "custom":
                if payload.get("type") == "token":
                    final_answer += payload.get("content", "")
                yield payload
                continue
            intent_update = payload.get("intent")
            if isinstance(intent_update, dict):
                track_plan = bool(intent_update.get("intent", {}).get("tasks"))
            tool_update = payload.get("tools")
            if isinstance(tool_update, dict):
                tool_messages = tool_update.get("messages", [])
                returned_tools.extend(message for message in tool_messages if isinstance(message, ToolMessage))
                tool_names = [getattr(message, "name", "工具") for message in tool_messages]
                frame = {"type": "tool_end", "tools": tool_names}
                urls = web_result_urls(tool_messages)
                new_urls = set()
                if urls is not None:
                    new_urls = urls - seen_web_urls
                    frame["web_result_count"] = len(new_urls)
                    seen_web_urls.update(urls)
                yield frame
                pages = {
                    page["url"]: page for message in tool_messages
                    if isinstance(message, ToolMessage) and message.status == "success" and isinstance(message.artifact, dict)
                    for page in message.artifact.get("read_pages", [])
                }
                if pages:
                    yield {"type": "web_pages_read", "pages": list(pages.values())}
                if new_urls:
                    highlights = search_highlights(question, tool_messages, new_urls)
                    if highlights:
                        yield {"type": "web_highlights", "pages": highlights}
    if run_id:
        snapshot = await app.aget_state(config)
        if snapshot.values.get("finished_run_id") != run_id:
            raise RuntimeError("任务缺少已完成的工作流检查点。")
        history = snapshot.values["messages"]
        yield _done_event(history[-1].content, history)
    else:
        yield _done_event(final_answer, returned_tools)


def _prepare_retry(history, messages, stable_id, question, retry):
    matching_index = next((i for i, message in enumerate(history) if message.id == stable_id), None)
    latest_question_index = next(
        (i for i in range(len(history) - 1, -1, -1) if isinstance(history[i], HumanMessage)),
        None,
    )
    if matching_index is not None:
        if not retry or matching_index != latest_question_index or history[matching_index].content != question:
            return {"type": "error", "message": "该问题已提交或不是当前会话的最后一轮，请发送新问题。", "retryable": False}
        messages.extend(RemoveMessage(id=message.id) for message in history[matching_index + 1:] if message.id)
    elif retry and latest_question_index is not None:
        latest = history[latest_question_index]
        if latest.id and not latest.id.startswith("web:") and latest.content == question:
            # 旧版客户端的最后一轮没有前端 ID，将它一次性换成新 ID。
            messages[:0] = [
                RemoveMessage(id=message.id)
                for message in history[latest_question_index:] if message.id
            ]
    return None
