"""根据检索证据决定下一步，并构造受控的检索调用。"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer

from backend.app.agent.context import (
    content_to_text,
    context_budget,
    messages_token_count,
    split_turns,
)
from backend.app.agent.evidence_retry import (
    evidence_retry_count,
    is_evidence_tool,
    parse_evidence_decision,
    retry_prompt,
    web_evidence_retry_limit,
    web_retry_made_progress,
    validated_query,
)
from backend.app.agent.telemetry import observe_async_operation
from backend.app.agent.workflow_state import AgentState
from backend.app.observability import event
from backend.app.prompts import prompt_text
from backend.app.quality.validator import HIGH_RISK_TERMS


def prepare_knowledge_lookup(state: AgentState):
    name = "search_knowledge_base"
    get_stream_writer()({"type": "tool_candidates", "count": 1, "tools": [name], "retrieval_mode": "explicit"})
    get_stream_writer()({"type": "tool_start", "tools": [name]})
    call = {"name": name, "id": f"knowledge-{uuid4().hex}", "args": {"query": state["intent"]["knowledge_query"]}}
    return {"messages": [AIMessage(content="", tool_calls=[call])], "allowed_tool_names": [name]}


def prepare_evidence_retry(state: AgentState):
    name = "search_knowledge_base" if state["intent"].get("knowledge_query") else "web_search"
    call = {"name": name, "id": f"retry-evidence-{uuid4().hex}", "args": {"query": state["evidence_retry_query"]}}
    writer = get_stream_writer()
    writer({"type": "tool_candidates", "count": 1, "tools": [name], "retrieval_mode": "evidence_retry"})
    writer({"type": "tool_start", "tools": [name]})
    return {
        "messages": [AIMessage(content="", tool_calls=[call])],
        "allowed_tool_names": [name], "draft_answer": None, "evidence_retry_query": None,
    }


async def review_evidence(
    state: AgentState, *,
    max_total, max_batch, tool_by_name, mcp_tool_names,
    chat_model,
):
    """在写正文前，根据本轮只读检索的结果决定下一步。"""
    intent = state["intent"]
    if intent.get("tasks") or intent.get("weather_assessment") or intent.get("semantic_direct"):
        return {}
    turn = split_turns(state["messages"])[-1]
    if not isinstance(turn[0], HumanMessage):
        return {}
    question = intent.get("resolved_question") or content_to_text(turn[0].content)
    if any(term in question.lower() for term in HIGH_RISK_TERMS):
        return {}
    knowledge = bool(intent.get("knowledge_query"))
    name = "search_knowledge_base" if knowledge else "web_search"
    results = [message for message in turn if isinstance(message, ToolMessage)]
    if not results or any(
        not is_evidence_tool(message.name, knowledge=knowledge)
        for message in results
    ):
        return {}
    retries = evidence_retry_count(state)
    retry_limit = 1 if knowledge else web_evidence_retry_limit(question)
    iterative_web = retry_limit > 1
    no_progress = iterative_web and retries > 0 and not web_retry_made_progress(question, turn)
    remaining = min(retry_limit - retries, max_total - state.get("tool_calls_used", 0))
    can_search = (
        remaining > 0 and not no_progress and max_batch > 0
        and name in tool_by_name and name not in mcp_tool_names
        and (knowledge or not intent.get("no_public_search"))
    )
    review_fallback = {"evidence_action": "partial"} if iterative_web and retries else {}
    invoke = getattr(chat_model(), "ainvoke", None)
    if not callable(invoke):
        return review_fallback
    coverage_instruction = "" if knowledge else (
        "按用户问题判断要交付的关键信息。问题写得短也可能需要更具体的明细。"
        "例如跨城市活动的举办地，城市只是概览，还应尽量找到主要活动的具体场馆，以及各阶段和地点的对应关系。"
        "已有概览但缺直接相关的明细时，可以定向搜索官方公告、指南、规格表或安排表。"
        "只补与问题直接相关的缺口；附加信息缺失时可直接回答。"
        '当 action 为 answer 或 partial 时，也可提取已有资料中最多三条读者自然会追问、且不重复主要明细的事实。'
        '例如主要场次的开赛时间、参观的入场安排、产品差异适合什么使用情境。'
        '用 details 数组返回，每项如 {"label":"开赛时间","text":"具体事实和时间口径","sources":[1]}，'
        'sources 使用支持该事实的摘录编号。用户明确要求简短时省略 details。'
        '当 action 为 answer 或 partial，且有至少两个有资料支持的项目共享字段时，请返回 '
        'table 对象，格式为 {"title":"明细的小标题","columns":["字段1","字段2"],'
        '"rows":[{"cells":["值1","值2"],"sources":[1]},{"cells":["值3","值4"],"sources":[2]}]}。'
        "sources 写摘录 S1、S2 中的数字。分阶段、分地点、版本对照等用这个字段；"
        "使用资料原有的正式阶段名和具体明细，避免把入围赛、瑞士轮泛化为首站、中段。不确定的单元格留空。"
    )
    prompt = [
        SystemMessage(
            prompt_text("evidence.review")
            + coverage_instruction
            + (
                f"还可补查 {remaining} 次。每次只查最值得补的缺口，已有足够证据就回答。"
                if can_search else "当前不能继续补查，请根据已有证据选择 answer、partial 或 clarify。"
            )
            + ("上一次补查没有增加可用正文。若还有缺口，保留已知结果并说明缺失项。" if no_progress else "")
        ),
        HumanMessage(retry_prompt(
            intent["knowledge_query"] if knowledge else question, "检查本轮证据是否足够",
            turn, knowledge=knowledge, excerpt_limit=1400, previous_gap=state.get("evidence_missing", ""),
        )),
    ]
    if messages_token_count(prompt) > context_budget():
        event("evidence_review_fallback", reason="context_budget", retries=retries)
        return review_fallback
    try:
        async with asyncio.timeout(6):
            async with observe_async_operation("evidence_review"):
                response = await invoke(prompt)
            decision = parse_evidence_decision(content_to_text(response.content), turn)
    except Exception as error:
        event("evidence_review_fallback", reason=type(error).__name__, retries=retries)
        return review_fallback
    action = decision.action
    stop_reason = None
    if action == "search" and not can_search:
        action = "partial"
    if action == "answer" and decision.missing:
        action = "partial"
    if action == "partial" and no_progress:
        stop_reason = "no_new_evidence"
    elif action == "partial" and remaining <= 0:
        stop_reason = "budget_exhausted"
    event(
        f"evidence_review_{action}", scope="knowledge" if knowledge else "web",
        retries=retries, stop_reason=stop_reason,
    )
    writer = get_stream_writer()
    writer({"type": "evidence_review", "action": action, **({"stop_reason": stop_reason} if stop_reason else {})})
    if action == "search":
        writer({"type": "evidence_retry", "reason": "已取得的资料缺少关键证据", "tool": name})
        return {"evidence_retry_query": decision.query, "evidence_retries": retries + 1,
                "evidence_action": None, "evidence_missing": decision.missing, "evidence_table": None, "evidence_details": []}
    if action == "clarify":
        return {"evidence_action": action, "intent": {**intent, "question": decision.clarification}}
    return {
        "evidence_action": action,
        "evidence_missing": decision.missing or (state.get("evidence_missing", "") if action == "partial" else ""),
        "evidence_table": decision.table.model_dump() if not knowledge and decision.table else None,
        "evidence_details": [detail.model_dump() for detail in decision.details] if not knowledge else [],
    }


async def retry_after_answer_check(
    state: AgentState, question, turn_messages, result, *,
    has_web_sources, chat_model, tool_by_name, mcp_tool_names, max_batch, max_total,
):
    """复核节点未能安排补查时，允许交付前按证据缺口再尝试一次。"""
    intent = state["intent"]
    knowledge = bool(intent.get("knowledge_query"))
    retry_tool = "search_knowledge_base" if knowledge else "web_search"
    evidence_results = [message for message in turn_messages if isinstance(message, ToolMessage)]
    scoped_results = evidence_results and all(
        is_evidence_tool(message.name, knowledge=knowledge) for message in evidence_results
    )
    missing_evidence = result["reason"] in {
        "没有可用的联网或天气工具证据", "没有可对应的知识库证据", "具体事件缺少本轮可引用的联网证据",
    }
    unconfirmed_coverage = bool(
        has_web_sources and intent.get("semantic_search") and state.get("evidence_action") in {None, "partial"}
    )
    if (
        ((not result["ok"] and (has_web_sources or missing_evidence)) or unconfirmed_coverage) and scoped_results
        and (knowledge or not intent.get("no_public_search"))
        and evidence_retry_count(state) == 0
        and retry_tool in tool_by_name and retry_tool not in mcp_tool_names
        and max_batch and state.get("tool_calls_used", 0) < max_total
        and not any(term in question.lower() for term in HIGH_RISK_TERMS)
    ):
        prompt = [
            SystemMessage(prompt_text("evidence.followup")),
            HumanMessage(retry_prompt(
                intent["knowledge_query"] if knowledge else question, result["reason"],
                turn_messages, knowledge=knowledge, excerpt_limit=1400,
            )),
        ]
        query = None
        if messages_token_count(prompt) <= context_budget():
            try:
                async with asyncio.timeout(6):
                    async with observe_async_operation("evidence_retry_decision"):
                        parts = [content_to_text(chunk.content) async for chunk in chat_model().astream(prompt)]
                    query = validated_query("".join(parts), turn_messages)
            except Exception as error:
                event("evidence_review_fallback", reason=type(error).__name__, retries=0)
        if query:
            get_stream_writer()({"type": "evidence_retry", "reason": "现有来源未能支持完整回答", "tool": retry_tool})
            return {
                "evidence_retry_query": query, "evidence_retries": 1, "evidence_action": None,
                "evidence_table": None, "evidence_details": [],
            }
    return {}
