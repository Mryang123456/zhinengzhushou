"""LangGraph 对话状态和检查点字段。"""

from __future__ import annotations

from typing import NotRequired

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """包含工具白名单及可恢复的上下文压缩状态。"""
    allowed_tool_names: NotRequired[list[str]]
    intent: NotRequired[dict]
    tool_calls_used: NotRequired[int]
    evidence_retry_used: NotRequired[bool]
    evidence_retries: NotRequired[int]
    evidence_retry_query: NotRequired[str | None]
    evidence_action: NotRequired[str | None]
    evidence_missing: NotRequired[str]
    evidence_table: NotRequired[dict | None]
    evidence_details: NotRequired[list[dict]]
    plan_outcomes: NotRequired[dict[int, str]]
    conversation_summary: NotRequired[str]
    summary_message_count: NotRequired[int]
    draft_answer: NotRequired[str | None]
    retry: NotRequired[bool]
    context_limit_reached: NotRequired[bool]
    durable_run_id: NotRequired[str | None]
    finished_run_id: NotRequired[str]
