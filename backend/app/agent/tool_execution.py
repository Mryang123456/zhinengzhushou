"""执行已登记工具，处理授权、审批与搜索结果预算。"""

from __future__ import annotations

import asyncio
import json
import os

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import ToolException
from langgraph.config import get_stream_writer

from backend.app.agent.compound_tasks import PLAN_WEB_RESULT_LIMIT
from backend.app.agent.context import content_to_text, split_turns
from backend.app.agent.evidence_retry import evidence_retry_count, web_evidence_retry_limit
from backend.app.agent.telemetry import observe_async_operation
from backend.app.agent.workflow_state import AgentState
from backend.app.intent.semantic import is_public_search_tool
from backend.app.mcp.governance import (
    RISK_ORDER,
    ToolPolicyError,
    audit,
    invoke_governed,
    risk_for_tool,
    sanitize_output,
    validate_tool_arguments,
)
from backend.app.quality.web_activity import search_queries
from backend.app.quality.web_fetch import enrich_search_results
from backend.app.quality.web_result_budget import WEB_RETRY_RESERVE, select_web_results


def sanitize_web_search_output(content: str) -> str:
    return sanitize_output(content, maximum=len(content))


async def execute_tools(
    state: AgentState, *,
    max_batch, max_total, user_id, tool_by_name,
    ensure_mcp_tools, recovering, retry_blocked_tools, mcp_tool_names,
    governance, get_approval_store, tool_node,
):
    """在执行前二次校验模型调用，阻止越权工具名。"""
    assistant_message = state["messages"][-1]
    calls = assistant_message.tool_calls

    def rejected(call, reason: str) -> ToolMessage:
        return ToolMessage(content=reason, tool_call_id=call["id"], name=call["name"], status="error")

    used = state.get("tool_calls_used", 0)
    if len(calls) > max_batch or used + len(calls) > max_total:
        audit({"user_id": user_id, "thread_id": "runtime", "tool": "batch", "status": "budget_blocked", "used": used, "requested": len(calls)})
        return {
            "messages": [rejected(call, "工具调用超出本回合预算，已停止执行并请用户缩小任务范围。") for call in calls],
            "tool_calls_used": min(max_total, used + len(calls)),
        }

    allowed = set(state.get("allowed_tool_names", []))
    if allowed - tool_by_name.keys():
        await ensure_mcp_tools()
    public_tools = {name for name in allowed & tool_by_name.keys() if is_public_search_tool(name)}
    if public_tools and not state["intent"].get("no_public_search"):
        queries = search_queries(calls, public_tools)
        if queries:
            get_stream_writer()({"type": "web_queries", "queries": queries})
    results: dict[str, ToolMessage] = {}
    local_calls = []
    for call in calls:
        name = call["name"]
        if state["intent"].get("no_public_search") and is_public_search_tool(name):
            results[call["id"]] = rejected(call, "用户要求本轮不搜索公开资料。")
            continue
        if (recovering or state.get("retry")) and name in retry_blocked_tools:
            results[call["id"]] = rejected(call, "重试只允许读取资料，已阻止重复修改记忆或发起审批。")
            continue
        if name not in allowed or name not in tool_by_name:
            results[call["id"]] = rejected(call, "该工具未获本轮授权或当前不可用，请改用可用工具或说明限制。")
            continue
        if name not in mcp_tool_names:
            local_calls.append(call)
            continue

        risk = risk_for_tool(name, governance)
        try:
            arguments = validate_tool_arguments(name, call["args"], governance)
        except ToolPolicyError as error:
            audit({"user_id": user_id, "thread_id": "runtime", "tool": name, "risk": risk, "status": "policy_blocked", "reason": str(error)})
            results[call["id"]] = rejected(call, f"工具调用已被策略阻止：{error}")
            continue
        if RISK_ORDER.get(risk, 4) >= RISK_ORDER["L2"]:
            approval_store = get_approval_store()
            assert approval_store is not None
            question = next((content_to_text(message.content) for message in reversed(state["messages"]) if isinstance(message, HumanMessage)), "")
            approval = approval_store.create(user_id, name, arguments, risk, question)
            audit({"user_id": user_id, "thread_id": "runtime", "tool": name, "risk": risk, "status": "approval_required", "approval_id": approval["approval_id"]})
            get_stream_writer()({"type": "approval_required", **approval})
            results[call["id"]] = rejected(call, "该操作需要你的明确确认。确认前不会调用工具或产生外部影响。")
        else:
            try:
                content = await invoke_governed(tool_by_name[name], arguments, risk, user_id, "runtime", governance.get("timeoutSeconds", 15), governance.get("retries", 1))
            except ToolException as error:
                results[call["id"]] = rejected(call, str(error))
            else:
                results[call["id"]] = ToolMessage(content=content, tool_call_id=call["id"], name=name)

    if local_calls:
        # ToolNode 只看到通过校验的本地调用，原始模型消息仍保留在 Checkpoint 中。
        filtered_message = assistant_message.model_copy(update={"tool_calls": local_calls})
        local_state = {**state, "messages": [*state["messages"][:-1], filtered_message]}
        try:
            async with observe_async_operation("tool_batch"):
                result = await asyncio.wait_for(tool_node.ainvoke(local_state), timeout=float(os.getenv("AGENT_TOOL_TIMEOUT_SECONDS", "20")))
            for message in result["messages"]:
                content = (
                    sanitize_web_search_output(message.content)
                    if message.name == "web_search" and isinstance(message.content, str)
                    else sanitize_output(message.content)
                )
                results[message.tool_call_id] = ToolMessage(
                    content=content, tool_call_id=message.tool_call_id,
                    name=message.name, status=message.status,
                )
        except TimeoutError:
            for call in local_calls:
                results[call["id"]] = rejected(call, "工具调用超时，已停止执行。")

    web_batch = [results[call["id"]] for call in calls if is_public_search_tool(call["name"])
                 and results[call["id"]].status == "success"]
    if web_batch:
        turn = split_turns(state["messages"])[-1]
        intent = state["intent"]
        question = intent.get("resolved_question") or content_to_text(turn[0].content)
        retries = evidence_retry_count(state)
        retry_limit = web_evidence_retry_limit(question)
        selected = select_web_results(
            turn, web_batch,
            reserve_retry=retries < retry_limit and not intent.get("tasks") and used + len(calls) < max_total,
            reserved_pages=WEB_RETRY_RESERVE if not retries else WEB_RETRY_RESERVE // 2,
            per_tool_limit=PLAN_WEB_RESULT_LIMIT if intent.get("tasks") else None,
        )
        for call in calls:
            call_id = call["id"]
            if call_id not in selected:
                continue
            content = json.dumps(selected[call_id], ensure_ascii=False)
            read_pages = []
            if call["name"] == "web_search" and selected[call_id]:
                content = sanitize_web_search_output(
                    await enrich_search_results(content, call["args"]["query"], read_pages=read_pages)
                )
            results[call_id] = results[call_id].model_copy(update={
                "content": content, "artifact": {"read_pages": read_pages} if read_pages else None,
            })

    return {
        "messages": [results[call["id"]] for call in calls],
        "tool_calls_used": used + len(calls),
    }
