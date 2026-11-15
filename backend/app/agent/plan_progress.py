"""从任务计划和已有工具消息生成可重放的进度快照。"""

from langchain_core.messages import AIMessage, HumanMessage

from backend.app.agent.compound_tasks import plan_dependency_status, planned_tool_results, planned_weather
from backend.app.agent.context import message_text
from backend.app.content_safety import sanitize_untrusted_text


def plan_progress(state: dict) -> dict | None:
    intent = state.get("intent", {})
    if not intent.get("tasks") or not state.get("messages"):
        return None
    messages = state["messages"]
    start = next((index for index in range(len(messages) - 1, -1, -1) if isinstance(messages[index], HumanMessage)), None)
    if start is None:
        return None
    turn = messages[start:]
    question = intent.get("resolved_question") or message_text(turn[0])
    tasks, plan_id = intent["tasks"], intent["plan_id"]
    results = planned_tool_results(turn)
    calls = {call["id"] for message in turn if isinstance(message, AIMessage) for call in message.tool_calls}
    outcomes = state.get("plan_outcomes", {})
    steps = []
    for index, task in enumerate(tasks):
        call_id = f"{plan_id}-{index}"
        retry_id = f"{call_id}-retry"
        result = results.get(call_id)
        status = outcomes.get(index, "pending")
        reason = ""
        if task["condition"] == "if_rain" and result is None:
            dependency = plan_dependency_status(question, tasks, index, plan_id, results)
            if dependency == "skipped":
                status, reason = "skipped", "预报未显示降雨"
            elif dependency == "blocked":
                status, reason = "blocked", "未取得可确认降雨的预报"
            elif dependency == "waiting" and status == "blocked":
                reason = "缺少前置天气结果"
        if status == "pending":
            if retry_id in calls and retry_id not in results:
                status = "supplementing"
            elif result is not None:
                if task["kind"] == "weather":
                    status = "complete" if planned_weather(question, task, result) else "blocked"
                else:
                    status = "reviewing"
            elif call_id in calls:
                status = "running"
        if status == "blocked" and not reason:
            reason = "未取得足够资料" if result else "未执行"
        steps.append({
            "id": index, "title": sanitize_untrusted_text(task["text"], 160),
            "status": status, **({"reason": reason} if reason else {}),
        })
    return {"type": "plan_progress", "steps": steps}
