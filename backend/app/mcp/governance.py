"""MCP 工具治理：风险、审计、输出清洗和调用韧性。"""

import asyncio
import hashlib
import json
import re
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException

from backend.app.config import PROJECT_ROOT
from backend.app.content_safety import TOOL_OUTPUT_MAX_CHARS, sanitize_untrusted_text

RISK_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}


class ToolPolicyError(ValueError):
    """工具未注册或调用参数违反注册策略。"""


class McpOutcomeUnknown(ToolException):
    """审批后的调用未取得可确认的结果，外部副作用可能已发生。"""


def load_governance(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    return config.get("mcpGovernance", {})


def validate_mcp_servers(config: dict) -> None:
    """服务命令、参数和传输必须与静态注册表一致。"""
    governance, servers = config.get("mcpGovernance", {}), config.get("mcpServers", {})
    policies = governance.get("serverPolicies", {})
    if set(servers) != set(policies) or set(servers) != set(governance.get("allowedServers", [])):
        raise ToolPolicyError("MCP 服务注册表与实际配置不一致。")
    require_sandbox = os.getenv("MCP_SANDBOX_REQUIRED", "false").lower() in {"1", "true", "yes"}
    for name, server in servers.items():
        policy = policies[name]
        if any(server.get(key) != policy.get(key) for key in ("command", "args", "transport")):
            raise ToolPolicyError(f"MCP 服务 {name} 的启动配置未获授权。")
        if server.get("transport") != "stdio":
            raise ToolPolicyError(f"MCP 服务 {name} 使用了不允许的传输方式。")
        if require_sandbox and policy.get("isolation") != "container":
            raise ToolPolicyError(f"生产环境要求 MCP 服务 {name} 使用容器沙箱。")


def load_mcp_connections(config_path: Path) -> dict:
    """加载显式 MCP 服务配置；绝不从聊天输入中接受服务启动命令。"""
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        connections = config["mcpServers"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"MCP 配置无效：{config_path}；原因：{error}") from error
    if not isinstance(connections, dict) or not connections:
        raise ValueError("mcpServers 必须是非空对象。")
    validate_mcp_servers(config)
    allowed_servers = set(config.get("mcpGovernance", {}).get("allowedServers", []))
    unexpected = set(connections) - allowed_servers
    if unexpected:
        raise ValueError(f"MCP 配置包含未注册服务：{', '.join(sorted(unexpected))}")
    return connections


def risk_for_tool(name: str, governance: dict) -> str:
    risks = governance.get("toolRisk", {})
    prefix = max((prefix for prefix in risks if name.startswith(prefix)), key=len, default=None)
    return risks[prefix] if prefix is not None else "L2"


def policy_for_tool(name: str, governance: dict) -> dict | None:
    """按最长前缀匹配工具注册策略，精确工具名优先于通配前缀。"""
    policies = governance.get("toolPolicies", {})
    matched = [prefix for prefix in policies if name == prefix or name.startswith(prefix)]
    if not matched:
        return None
    return policies[max(matched, key=len)]


def is_registered_tool(name: str, governance: dict) -> bool:
    return policy_for_tool(name, governance) is not None


def validate_tool_arguments(name: str, arguments: dict, governance: dict) -> dict:
    """拒绝策略外参数，并对类型、长度和数值边界做严格校验。"""
    policy = policy_for_tool(name, governance)
    if policy is None:
        raise ToolPolicyError(f"MCP 工具未登记：{name}")
    if not isinstance(arguments, dict):
        raise ToolPolicyError("MCP 工具参数必须是对象。")
    definitions = policy.get("arguments", {})
    unexpected = set(arguments) - set(definitions)
    if unexpected:
        raise ToolPolicyError(f"包含未允许参数：{', '.join(sorted(unexpected))}")
    for key, definition in definitions.items():
        if definition.get("required") and key not in arguments:
            raise ToolPolicyError(f"缺少必填参数：{key}")
    checked: dict = {}
    for key, value in arguments.items():
        definition = definitions[key]
        expected_type = definition.get("type")
        if expected_type == "string":
            if not isinstance(value, str):
                raise ToolPolicyError(f"参数 {key} 必须是文本。")
            if len(value) < definition.get("minLength", 0) or len(value) > definition.get("maxLength", 4096):
                raise ToolPolicyError(f"参数 {key} 的长度不符合限制。")
        elif expected_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                raise ToolPolicyError(f"参数 {key} 必须是整数。")
        elif expected_type == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ToolPolicyError(f"参数 {key} 必须是数字。")
        elif expected_type == "boolean" and not isinstance(value, bool):
            raise ToolPolicyError(f"参数 {key} 必须是布尔值。")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value < definition.get("minimum", float("-inf")) or value > definition.get("maximum", float("inf")):
                raise ToolPolicyError(f"参数 {key} 超出允许范围。")
        checked[key] = value
    return checked


def sanitize_output(value: object, maximum: int = TOOL_OUTPUT_MAX_CHARS) -> str:
    return sanitize_untrusted_text(value, maximum)


def audit(event: dict) -> None:
    path = PROJECT_ROOT / "data" / "mcp_audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = {**event, "at": datetime.now(timezone.utc).isoformat()}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(safe, ensure_ascii=False) + "\n")


async def invoke_governed(tool, arguments: dict, risk: str, user_id: str, thread_id: str, timeout_seconds: int = 15, retries: int = 1, approved: bool = False, approval_id: str | None = None) -> str:
    """L2 及以上默认阻断；L0/L1 具备超时、重试、审计和输出清洗。"""
    argument_hash = hashlib.sha256(json.dumps(arguments, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    requires_approval = RISK_ORDER.get(risk, 4) >= RISK_ORDER["L2"]
    if requires_approval and not approved:
        audit({"user_id": user_id, "thread_id": thread_id, "tool": tool.name, "risk": risk, "args_hash": argument_hash, "status": "approval_required", "approval_id": approval_id})
        raise ToolException("该工具涉及私有读取或写入操作，必须经过用户确认后才能执行。")
    retries = 0 if requires_approval else max(0, retries)
    for attempt in range(retries + 1):
        try:
            result = await asyncio.wait_for(tool.ainvoke(arguments), timeout=timeout_seconds)
            if isinstance(result, ToolMessage):
                if result.status != "success":
                    raise ToolException("MCP 工具返回失败状态。")
                result = result.content
            if isinstance(result, list) and result and all(
                isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
                for block in result
            ):
                result = "\n".join(block["text"] for block in result)
            if result is None or result == [] or result == {} or (isinstance(result, str) and not result.strip()):
                raise ToolException("MCP 工具没有返回内容。")
            audit({"user_id": user_id, "thread_id": thread_id, "tool": tool.name, "risk": risk, "args_hash": argument_hash, "status": "success", "attempt": attempt + 1, "approval_id": approval_id})
            return sanitize_output(result)
        except Exception as error:
            if attempt == retries:
                status = "outcome_unknown" if requires_approval else "failed"
                audit({"user_id": user_id, "thread_id": thread_id, "tool": tool.name, "risk": risk, "args_hash": argument_hash, "status": status, "error": type(error).__name__, "approval_id": approval_id})
                if requires_approval:
                    raise McpOutcomeUnknown("无法确认已审批操作的执行结果。请先核对外部系统状态，再决定是否重新提交。") from error
                raise ToolException("MCP 工具调用失败、超时或没有返回内容。") from error
    raise ToolException("MCP 工具调用失败。")
