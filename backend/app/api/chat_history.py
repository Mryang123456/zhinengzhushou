"""将已完成的聊天任务转换成客户端可以恢复的会话。"""

PUBLIC_ACTIVITY = {
    "progress": ("stage",),
    "plan_progress": ("steps",),
    "tool_start": ("tools",),
    "tool_end": ("tools", "web_result_count"),
    "evidence_retry": (),
    "evidence_review": ("action", "stop_reason"),
    "quality": ("passed", "partial"),
    "web_pages_read": ("pages",),
    "web_queries": ("queries",),
    "web_highlights": ("pages",),
    "timing": ("operation", "elapsed_ms"),
}


def serialize_chat_history(rows) -> list[dict]:
    sessions = {}
    for thread, message, question, finished_at, answer, sources, run_id, events, elapsed_ms in rows:
        if not isinstance(question, str) or not isinstance(answer, str):
            continue
        title = " ".join(question.split())[:24] or "同步会话"
        session = sessions.setdefault(thread, {"id": thread, "title": title, "messages": [], "updatedAt": 0})
        reply = {"id": f"{message}:answer", "role": "assistant", "content": answer, "completedRunId": str(run_id),
                 "completedAt": int(finished_at.timestamp() * 1000) if finished_at else 0}
        if sources:
            reply["sources"] = sources
        if elapsed_ms is not None:
            reply["elapsedMs"] = elapsed_ms
        activity = []
        for event in events or []:
            kind = event.get("type")
            if kind in PUBLIC_ACTIVITY:
                activity.append({"type": kind, **{field: event[field] for field in PUBLIC_ACTIVITY[kind] if field in event}})
        if activity:
            reply["activity"] = activity
        session["messages"].extend(({"id": str(message), "role": "user", "content": question}, reply))
        session["updatedAt"] = max(session["updatedAt"], reply["completedAt"])
    return list(sessions.values())
