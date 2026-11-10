"""管理对话检查点、请求重试与流式回合。"""

from contextlib import aclosing, asynccontextmanager

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver


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
        {"messages": [HumanMessage(question)], "retry": False},
        config=checkpoint_config(thread_id, user_id),
    )
    return result["messages"][-1].content


async def stream_events(
    app, question: str, thread_id: str, user_id: str = "default", *,
    message_id: str | None = None, retry: bool = False,
):
    """逐步产出 token、工具状态和最终答案，供命令行或 SSE 接口复用。"""
    config = checkpoint_config(thread_id, user_id)
    stable_id = f"web:{message_id}" if message_id is not None else None
    question_message = HumanMessage(content=question, id=stable_id)
    messages: list[BaseMessage] = [question_message]
    if stable_id is not None:
        snapshot = await app.aget_state(config)
        history = snapshot.values.get("messages", [])
        matching_index = next((i for i, message in enumerate(history) if message.id == stable_id), None)
        latest_question_index = next(
            (i for i in range(len(history) - 1, -1, -1) if isinstance(history[i], HumanMessage)),
            None,
        )
        if matching_index is not None:
            if not retry or matching_index != latest_question_index or history[matching_index].content != question:
                yield {"type": "error", "message": "该问题已提交或不是当前会话的最后一轮，请发送新问题。", "retryable": False}
                return
            # 按 ID 更新原问题，同时清理这次尝试产生的草稿与工具调用记录。
            messages.extend(RemoveMessage(id=message.id) for message in history[matching_index + 1:] if message.id)
        elif retry and latest_question_index is not None:
            latest = history[latest_question_index]
            if latest.id and not latest.id.startswith("web:") and latest.content == question:
                # 旧版客户端的最后一轮没有前端 ID，将它一次性换成新 ID。
                messages = [
                    RemoveMessage(id=message.id)
                    for message in history[latest_question_index:] if message.id
                ] + [question_message]
    final_answer = ""
    async with aclosing(app.astream(
        {"messages": messages, "retry": retry},
        config=config,
        stream_mode=["custom", "updates"],
    )) as events:
        async for mode, payload in events:
            if mode == "custom":
                if payload.get("type") == "token":
                    final_answer += payload.get("content", "")
                yield payload
                continue
            tool_update = payload.get("tools")
            if isinstance(tool_update, dict):
                tool_names = [getattr(message, "name", "工具") for message in tool_update.get("messages", [])]
                yield {"type": "tool_end", "tools": tool_names}
    yield {"type": "done", "content": final_answer}
