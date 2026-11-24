"""识别当前请求、解析追问并生成有依赖的只读任务。"""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.config import get_stream_writer

from backend.app.agent.context import content_to_text, incomplete_tool_message_ids, split_turns
from backend.app.agent.telemetry import observe_async_operation
from backend.app.prompts import prompt_text
from backend.app.agent.workflow_state import AgentState
from backend.app.intent.capabilities import describe_capabilities
from backend.app.intent.followup import resolve_weather_followup
from backend.app.intent.planning import (
    IntentPlan,
    RAIN_CONDITION,
    explicit_weather_plan,
    planning_candidate,
    validate_plan,
)
from backend.app.intent.policy import classify, is_verification_followup, requested_knowledge_query
from backend.app.intent.semantic import NO_WEB, resolve_intent, without_no_web_clause
from backend.app.quality.validator import HIGH_RISK_TERMS


logger = logging.getLogger(__name__)


async def identify_intent(
    state: AgentState, *,
    initial_intent, initial_question, skills, tool_by_name,
    max_batch, max_total, mcp_tool_names, governance,
    recovering, retry_blocked_tools, knowledge_database_url, chat_model,
):
    get_stream_writer()({"type": "progress", "stage": "understanding"})
    question = next((content_to_text(message.content) for message in reversed(state["messages"]) if isinstance(message, HumanMessage)), "")
    original_question = question
    no_public_search = bool(NO_WEB.search(original_question))
    question = resolve_weather_followup(question, state["messages"], state.get("intent") or {})
    decision = initial_intent if initial_intent is not None and question == initial_question else classify(question, skills)
    if no_public_search and decision.get("general"):
        subject_policy = classify(without_no_web_clause(question), skills)
        if subject_policy.get("requires_web_evidence"):
            decision = subject_policy
    ambiguous_verification = False
    if is_verification_followup(question):
        history = split_turns(state["messages"])[-7:-1]
        previous = state.get("intent") or {}
        candidates = {}
        for turn in history:
            if not isinstance(turn[0], HumanMessage) or not isinstance(turn[-1], AIMessage) or turn[-1].tool_calls:
                continue
            topic = content_to_text(turn[0].content)
            if turn is history[-1] and is_verification_followup(topic):
                topic = previous.get("resolved_question") or topic
            prior = classify(topic, skills)
            if prior.get("requires_web_evidence") and prior.get("web_query"):
                candidates[prior["web_query"]] = (topic, prior)
        if len(candidates) == 1:
            question, prior = next(iter(candidates.values()))
            decision = {**prior, "reason": "继续核实上一轮明确的事件" if len(history) == 1 else "继续核实近期唯一明确的事件"}
        elif len(candidates) > 1:
            ambiguous_verification = True
            decision = {**decision, "reason": "近期有多个待核实事件",
                        "question": "刚才聊到了不止一件事。你想查证其中哪一件？"}
    weather_plan = not no_public_search and explicit_weather_plan(question)
    verification_blocked = decision.get("requires_web_evidence") and (
        no_public_search or "web_search" not in tool_by_name or not max_batch or not max_total
    )
    if (
        requested_knowledge_query(original_question) is None
        and question == original_question
        and not ambiguous_verification
        and not weather_plan
        and not verification_blocked
        and not decision.get("direct_reply")
        and decision.get("reason") != "基础任务直通"
    ):
        capabilities = describe_capabilities(
            tool_by_name, mcp_tool_names, governance,
            blocked_tools=retry_blocked_tools if recovering or state.get("retry") else set(),
            max_total=max_total, max_batch=max_batch,
            public_search_allowed=not no_public_search,
            knowledge_configured=bool(knowledge_database_url),
        )
        async with observe_async_operation("intent_resolution"):
            semantic = await resolve_intent(
                chat_model(), original_question, state["messages"][:-1], capabilities=capabilities,
            )
        if semantic:
            rewritten = semantic.question.strip()
            rewritten_policy = classify(rewritten, skills)
            if requested_knowledge_query(rewritten) is None and not any(
                term in original_question and term not in rewritten for term in HIGH_RISK_TERMS
            ):
                if semantic.route == "clarify":
                    decision = {
                        **decision, "status": "clarify", "reason": "语义识别发现缺少关键信息",
                        "question": semantic.clarification.strip(),
                    }
                elif not decision.get("weather_assessment") and semantic.route == "search" and not (
                    not capabilities["public_search"] or NO_WEB.search(rewritten)
                    or rewritten_policy.get("weather_assessment")
                    or planning_candidate(rewritten)
                ):
                    question = rewritten
                    decision = {
                        **decision,
                        "status": "allow", "reason": "语义识别需要联网查证", "skills": [],
                        "confidence": 0.0, "question": "", "requires_web_evidence": True,
                        "web_query": semantic.query.strip(), "web_queries": semantic.queries,
                        "semantic_search": True,
                    }
                elif not decision.get("weather_assessment") and rewritten_policy.get("weather_assessment") and semantic.route in {"tools", "search"}:
                    question = rewritten
                    decision = {**rewritten_policy, "reason": "语义识别需要天气工具"}
                elif not decision.get("weather_assessment") and semantic.route == "tools" and (
                    planning_candidate(rewritten)
                    or rewritten_policy.get("skills")
                    or (decision.get("general") and rewritten_policy.get("general"))
                ):
                    question = rewritten
                    decision = {**rewritten_policy, "general": False, "semantic_tools": True, "reason": "语义识别需要工具"}
                elif semantic.route == "direct" and decision.get("general") and not (
                    rewritten_policy.get("requires_web_evidence") or planning_candidate(rewritten)
                ):
                    question = rewritten
                    decision = {**decision, "reason": "语义识别可直接回答", "semantic_direct": True}
    if no_public_search:
        if decision.get("requires_web_evidence"):
            decision = {
                **decision, "status": "clarify", "reason": "需要联网核实但用户禁止联网",
                "question": "这件事需要来源才能核实。你可以发给我已有的资料，我会基于你提供的内容分析。",
            }
        decision = {**decision, "no_public_search": True}
    if decision["status"] == "allow" and decision.get("requires_web_evidence") and (
        "web_search" not in tool_by_name or not max_batch or not max_total
    ):
        decision = {
            **decision, "status": "clarify", "reason": "联网核实不可用",
            "question": "当前无法联网核实。你可以提供公告或报道，我会基于你提供的内容分析。",
        }
    if (
        (decision["status"] != "clarify" or weather_plan)
        and not decision.get("semantic_direct")
        and not no_public_search
        and decision.get("knowledge_query") is None
        and not decision.get("requires_web_evidence")
        and not any(term in question for term in HIGH_RISK_TERMS)
        and planning_candidate(question)
    ):
        schema = json.dumps(IntentPlan.model_json_schema(), ensure_ascii=False)
        prompt = SystemMessage(prompt_text("intent.plan") + "\nJSON Schema：" + schema)
        try:
            async with asyncio.timeout(8):
                async with observe_async_operation("intent_planning"):
                    parts = [content_to_text(chunk.content) async for chunk in chat_model().astream([prompt, HumanMessage(question)])]
                tasks = validate_plan(IntentPlan.model_validate_json("".join(parts)), question)
        except Exception as error:
            logger.info("结构化意图识别回退：%s", type(error).__name__)
            tasks = []
        if tasks and (not max_batch or len(tasks) > max_total):
            decision = {**decision, "status": "clarify", "reason": "复合任务超出工具预算",
                        "question": "当前工具额度不足以处理完整请求。请缩小范围后重试。"}
        elif tasks:
            decision = {**decision, "status": "allow", "reason": "已拆分复合只读任务",
                        "tasks": tasks, "plan_id": f"intent-{uuid4().hex}"}
        elif RAIN_CONDITION.search(question):
            decision = {**decision, "status": "clarify", "reason": "未能安全拆解天气条件任务",
                        "question": "我还没能明确拆分这个条件任务。请告诉我需要查询的城市和日期，以及下雨时要我做什么。"}
        elif weather_plan:
            decision = {**decision, "status": "clarify", "reason": "未能安全拆解多项天气查询",
                        "question": "请分别列出每一项要查的城市和日期，我会逐项查询。"}
    if decision.get("knowledge_query") and ("search_knowledge_base" not in tool_by_name or not max_total or not max_batch):
        decision = {**decision, "status": "clarify", "reason": "知识库检索不可用", "question": "当前无法检索知识库。请确认服务已配置并启用知识库检索后重试。"}
    if original_question != question:
        decision = {**decision, "resolved_question": question}
    get_stream_writer()({"type": "intent", "status": decision["status"], "confidence": decision["confidence"], "reason": decision["reason"]})
    update = {"intent": decision, "tool_calls_used": 0, "draft_answer": None, "context_limit_reached": False,
              "evidence_retries": 0, "evidence_retry_query": None, "evidence_action": None,
              "evidence_missing": "", "evidence_table": None, "evidence_details": [], "plan_outcomes": {}}
    messages = state["messages"]
    discarded = incomplete_tool_message_ids(messages)
    if "draft_answer" not in state:
        # 旧质量门在草稿之后追加降级回答，形成相邻的两条无工具调用的助手消息。
        discarded.update(
            earlier.id
            for earlier, later in zip(messages, messages[1:])
            if isinstance(earlier, AIMessage) and isinstance(later, AIMessage)
            and earlier.id and not earlier.tool_calls and not later.tool_calls
        )
    if discarded:
        update.update(
            messages=[RemoveMessage(id=message.id) for message in messages if message.id in discarded],
            conversation_summary="",
            summary_message_count=0,
        )
    return update
