"""首次路由所需的后端能力摘要。"""

from collections.abc import Mapping, Set

from backend.app.content_safety import sanitize_untrusted_text
from backend.app.mcp.governance import RISK_ORDER, risk_for_tool


def describe_capabilities(
    tools: Mapping[str, object],
    mcp_names: Set[str],
    governance: dict,
    *,
    blocked_tools: Set[str],
    max_total: int,
    max_batch: int,
    public_search_allowed: bool,
    knowledge_configured: bool,
) -> dict:
    """只描述已注册能力。具体请求的授权由执行链路判断。"""
    batch_limit = min(max_batch, max_total)
    available = {}
    if batch_limit > 0:
        for name, tool in tools.items():
            public_search = name == "web_search" or (name.startswith("search_") and name != "search_knowledge_base")
            if name in blocked_tools or (public_search and not public_search_allowed):
                continue
            if name in mcp_names:
                requires_approval = RISK_ORDER.get(risk_for_tool(name, governance), 4) >= RISK_ORDER["L2"]
                access = "requires_approval" if requires_approval else "governed_mcp"
            elif name in {"remember_memory", "forget_memory"}:
                access = "writes_memory"
            else:
                access = "read_only"
            # MCP 描述来自外部服务。摘要只暴露工具名称和后端确定的策略。
            available[name] = {
                "access": access,
                "description": "" if name in mcp_names else sanitize_untrusted_text(getattr(tool, "description", ""), 160),
            }

    return {
        "tools": [{"name": name, **details} for name, details in available.items()][:24],
        "knowledge_base": {
            "configured": knowledge_configured,
            "registered": "search_knowledge_base" in tools,
            "available_this_turn": "search_knowledge_base" in available,
            "requires_explicit_request": True,
        },
        "public_search": "web_search" in available,
        "remaining_tool_calls": max_total,
        "max_calls_per_batch": batch_limit,
    }
