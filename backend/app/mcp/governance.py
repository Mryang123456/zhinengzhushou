"""MCP 工具治理：风险、审计、输出清洗和调用韧性。"""

import asyncio
import hashlib
import json
import re
import os
from datetime import datetime, timezone
from pathlib import Path

from backend.app.config import PROJECT_ROOT

RISK_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}
INJECTION = re.compile(r"ignore (previous|all) instructions|system prompt|你现在是|忽略.*指令", re.I)


class ToolPolicyError(ValueError):
    """工具未注册或调用参数违反注册策略。"""


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


def risk_for_tool(name: str, governance: dict) -> str:
    for prefix, risk in governance.get("toolRisk", {}).items():
        if name.startswith(prefix):
            return risk
    return "L2"


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


def sanitize_output(value: object, maximum: int = 12000) -> str:
    text = str(value)[:maximum]
    return INJECTION.sub("[已移除疑似提示词注入内容]", text)


def audit(event: dict) -> None:
    path = PROJECT_ROOT / "data" / "mcp_audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = {**event, "at": datetime.now(timezone.utc).isoformat()}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(safe, ensure_ascii=False) + "\n")


async def invoke_governed(tool, arguments: dict, risk: str, user_id: str, thread_id: str, timeout_seconds: int = 15, retries: int = 1, approved: bool = False, approval_id: str | None = None) -> str:
    """L2 及以上默认阻断；L0/L1 具备超时、重试、审计和输出清洗。"""
    argument_hash = hashlib.sha256(json.dumps(arguments, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    if RISK_ORDER.get(risk, 4) >= RISK_ORDER["L2"] and not approved:
        audit({"user_id": user_id, "thread_id": thread_id, "tool": tool.name, "risk": risk, "args_hash": argument_hash, "status": "approval_required", "approval_id": approval_id})
        return "该工具涉及私有读取或写入操作，必须经过用户确认后才能执行。"
    for attempt in range(retries + 1):
        try:
            result = await asyncio.wait_for(tool.ainvoke(arguments), timeout=timeout_seconds)
            audit({"user_id": user_id, "thread_id": thread_id, "tool": tool.name, "risk": risk, "args_hash": argument_hash, "status": "success", "attempt": attempt + 1, "approval_id": approval_id})
            return sanitize_output(result)
        except Exception as error:
            if attempt == retries:
                audit({"user_id": user_id, "thread_id": thread_id, "tool": tool.name, "risk": risk, "args_hash": argument_hash, "status": "failed", "error": type(error).__name__, "approval_id": approval_id})
                return "工具调用失败或超时，已停止重试。"
    return "工具调用失败。"
