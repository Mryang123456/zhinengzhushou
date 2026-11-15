"""装配 Agent 依赖并注册 LangGraph 工作流。"""

from __future__ import annotations

import asyncio
from functools import partial
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import ToolException
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from backend.app.agent.answer_node import (
    clarify_intent,
    emit_approved_answer,
    quality_gate,
    reply_to_greeting,
)
from backend.app.agent.context import compact_context, content_to_text
from backend.app.agent.evidence_node import (
    prepare_evidence_retry,
    prepare_knowledge_lookup,
    review_evidence,
)
from backend.app.agent.intent_node import identify_intent
from backend.app.agent.model_node import SYSTEM_PROMPT, call_model
from backend.app.agent.tool_execution import (
    execute_tools,
    sanitize_web_search_output as sanitize_web_search_output,
)
from backend.app.agent.tools import (
    TOOLS,
    calculator as calculator,
    create_knowledge_tools,
    create_memory_tools,
    create_skill_tools,
    evaluate_expression as evaluate_expression,
    get_weather as get_weather,
    weather_code_description as weather_code_description,
    web_search as web_search,
)
from backend.app.agent.turns import (
    ask as ask,
    checkpoint_config as checkpoint_config,
    checkpoint_scope as checkpoint_scope,
    stream_events as stream_events,
)
from backend.app.agent.workflow_state import AgentState
from backend.app.config import (
    ENV_FILE,
    MCP_APPROVAL_DB_FILE,
    MCP_CONFIG_FILE,
    MEMORY_DB_FILE,
    SKILLS_DIR,
)
from backend.app.intent.policy import classify
from backend.app.mcp.approval import create_approval_store
from backend.app.mcp.discovery import discover_mcp_tools
from backend.app.mcp.governance import RISK_ORDER, load_governance, risk_for_tool, sanitize_output
from backend.app.memory.store import create_memory_store
from backend.app.models.factory import create_chat_model
from backend.app.quality.validator import validate_answer
from backend.app.skills.registry import load_skills


load_dotenv(ENV_FILE)


async def build_agent(
    mcp_config_path: Path | None = None,
    user_id: str = "default",
    memory_db_path: Path | None = MEMORY_DB_FILE,
    memory_database_url: str | None = None,
    checkpointer=None,
    *,
    initial_question: str | None = None,
    recovering: bool = False,
):
    """构建同时包含本地工具和已发现 MCP 工具的工作流。"""
    tools = TOOLS.copy()
    mcp_tool_names: set[str] = set()
    governance = load_governance(mcp_config_path or MCP_CONFIG_FILE)
    max_total = max(0, int(governance.get("maxToolCalls", os.getenv("AGENT_MAX_TOOL_CALLS", "6"))))
    max_batch = max(0, int(governance.get("maxToolBatch", os.getenv("AGENT_MAX_TOOL_BATCH", "3"))))
    skills = load_skills(SKILLS_DIR)
    initial_intent = classify(initial_question, skills) if initial_question is not None else None
    general = initial_intent is not None and initial_intent.get("general", False)
    knowledge_database_url = os.getenv("KNOWLEDGE_DATABASE_URL")
    tools.extend(create_skill_tools(skills))
    if initial_intent is None or initial_intent.get("skills") or initial_intent.get("knowledge_query"):
        tools.extend(create_knowledge_tools(knowledge_database_url, user_id))
    memory_store = None if general or (initial_intent or {}).get("direct_reply") else create_memory_store(memory_database_url, memory_db_path)
    if memory_store:
        tools.extend(create_memory_tools(memory_store, user_id))
    tool_by_name = {item.name: item for item in tools}
    retry_blocked_tools = {"remember_memory", "forget_memory"}
    approval_store = None
    discovery_attempted = False
    discovery_lock = asyncio.Lock()

    async def ensure_mcp_tools():
        nonlocal discovery_attempted, approval_store
        if mcp_config_path is None:
            return
        async with discovery_lock:
            if discovery_attempted:
                return
            discovered = await discover_mcp_tools(mcp_config_path, governance)
            for item in discovered:
                if item.name in tool_by_name:
                    continue
                tool_by_name[item.name] = item
                mcp_tool_names.add(item.name)
                if RISK_ORDER.get(risk_for_tool(item.name, governance), 4) >= RISK_ORDER["L2"]:
                    retry_blocked_tools.add(item.name)
            approval_store = create_approval_store(MCP_APPROVAL_DB_FILE) if mcp_tool_names else None
            discovery_attempted = True

    if initial_intent is None or (initial_intent["status"] == "allow" and initial_intent["skills"]):
        await ensure_mcp_tools()

    model = None

    def chat_model():
        nonlocal model
        if model is None:
            model = create_chat_model()
        return model

    def local_tool_error(error: Exception) -> str:
        return sanitize_output(str(error)) if isinstance(error, ToolException) else "本地工具执行失败。"

    tool_node = ToolNode(tools, handle_tool_errors=local_tool_error)

    graph = StateGraph(AgentState)
    graph.add_node("intent", partial(
        identify_intent,
        initial_intent=initial_intent,
        initial_question=initial_question,
        skills=skills,
        tool_by_name=tool_by_name,
        max_batch=max_batch,
        max_total=max_total,
        mcp_tool_names=mcp_tool_names,
        governance=governance,
        recovering=recovering,
        retry_blocked_tools=retry_blocked_tools,
        knowledge_database_url=knowledge_database_url,
        chat_model=chat_model,
    ))
    graph.add_node("clarify", clarify_intent)
    graph.add_node("direct_reply", reply_to_greeting)
    graph.add_node("knowledge", prepare_knowledge_lookup)
    graph.add_node("context", partial(compact_context, chat_model=chat_model))
    graph.add_node("agent", partial(
        call_model,
        memory_store=memory_store,
        user_id=user_id,
        chat_model=chat_model,
        max_batch=max_batch,
        max_total=max_total,
        skills=skills,
        tool_by_name=tool_by_name,
        ensure_mcp_tools=ensure_mcp_tools,
        mcp_tool_names=mcp_tool_names,
        recovering=recovering,
        retry_blocked_tools=retry_blocked_tools,
    ))
    graph.add_node("tools", partial(
        execute_tools,
        max_batch=max_batch,
        max_total=max_total,
        user_id=user_id,
        tool_by_name=tool_by_name,
        ensure_mcp_tools=ensure_mcp_tools,
        recovering=recovering,
        retry_blocked_tools=retry_blocked_tools,
        mcp_tool_names=mcp_tool_names,
        governance=governance,
        get_approval_store=lambda: approval_store,
        tool_node=tool_node,
    ))
    graph.add_node("review_evidence", partial(
        review_evidence,
        max_total=max_total,
        max_batch=max_batch,
        tool_by_name=tool_by_name,
        mcp_tool_names=mcp_tool_names,
        chat_model=chat_model,
    ))
    graph.add_node("quality", partial(
        quality_gate,
        chat_model=chat_model,
        tool_by_name=tool_by_name,
        mcp_tool_names=mcp_tool_names,
        max_batch=max_batch,
        max_total=max_total,
    ))
    graph.add_node("evidence_retry", prepare_evidence_retry)
    graph.add_node("output", emit_approved_answer)
    connect_workflow(graph)
    # 为工具批次、结果复核和最终回答预留图节点预算。
    return graph.compile(checkpointer=checkpointer or MemorySaver()).with_config(recursion_limit=4 * max_total + 7)


def route_intent(state: AgentState):
    intent = state["intent"]
    if intent["status"] == "clarify":
        return "clarify"
    if intent.get("direct_reply"):
        return "direct_reply"
    return "knowledge" if intent.get("knowledge_query") else "context"


def route_after_agent(state: AgentState):
    return "quality" if state.get("draft_answer") is not None else "tools"


def connect_workflow(graph: StateGraph) -> None:
    """工作流的节点名称和路由关系也用于恢复检查点。"""
    graph.add_edge(START, "intent")
    graph.add_conditional_edges("intent", route_intent, {
        "knowledge": "knowledge", "context": "context", "clarify": "clarify", "direct_reply": "direct_reply",
    })
    graph.add_edge("knowledge", "tools")
    graph.add_edge("context", "agent")
    graph.add_edge("clarify", "output")
    graph.add_edge("direct_reply", "output")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "quality": "quality"})
    graph.add_edge("tools", "review_evidence")
    graph.add_conditional_edges(
        "review_evidence",
        lambda state: "evidence_retry" if state.get("evidence_retry_query") else (
            "clarify" if state.get("evidence_action") == "clarify" else "context"
        ),
        {"evidence_retry": "evidence_retry", "clarify": "clarify", "context": "context"},
    )
    graph.add_conditional_edges(
        "quality", lambda state: "evidence_retry" if state.get("evidence_retry_query") else "output",
        {"evidence_retry": "evidence_retry", "output": "output"},
    )
    graph.add_edge("evidence_retry", "tools")
    graph.add_edge("output", END)


async def continue_after_approved_mcp(question: str, tool_name: str, tool_result: str) -> str:
    """将人工确认后的工具结果重新经过受控续答与质量门，不直接把原始输出给用户。"""
    model = create_chat_model()
    response = await model.ainvoke([
        SystemMessage(SYSTEM_PROMPT + "\n你只可依据给出的已确认工具结果回答原始问题；如证据不足请明确说明。"),
        HumanMessage(f"原始问题：{question}\n\n已确认工具：{tool_name}\n工具结果：\n{tool_result}"),
    ])
    answer = content_to_text(response.content)
    validation = validate_answer(question, answer, [ToolMessage(content=tool_result, tool_call_id="approved", name=tool_name)])
    return answer if validation["ok"] else validation["replacement"]


def main() -> None:
    """兼容直接运行本模块的命令行入口。"""
    from backend.app.agent.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
