"""MCP 工具治理：风险、审计、输出清洗和调用韧性。"""

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from backend.app.config import PROJECT_ROOT

RISK_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}
INJECTION = re.compile(r"ignore (previous|all) instructions|system prompt|你现在是|忽略.*指令", re.I)


def load_governance(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    return config.get("mcpGovernance", {})


def risk_for_tool(name: str, governance: dict) -> str:
    for prefix, risk in governance.get("toolRisk", {}).items():
        if name.startswith(prefix):
            return risk
    return "L2"


def sanitize_output(value: object, maximum: int = 12000) -> str:
    text = str(value)[:maximum]
    return INJECTION.sub("[已移除疑似提示词注入内容]", text)


def audit(event: dict) -> None:
    path = PROJECT_ROOT / "data" / "mcp_audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = {**event, "at": datetime.now(timezone.utc).isoformat()}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(safe, ensure_ascii=False) + "\n")


async def invoke_governed(tool, arguments: dict, risk: str, user_id: str, thread_id: str, timeout_seconds: int = 15, retries: int = 1) -> str:
    """L2 及以上默认阻断；L0/L1 具备超时、重试、审计和输出清洗。"""
    argument_hash = hashlib.sha256(json.dumps(arguments, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    if RISK_ORDER.get(risk, 4) >= RISK_ORDER["L2"]:
        audit({"user_id": user_id, "thread_id": thread_id, "tool": tool.name, "risk": risk, "args_hash": argument_hash, "status": "approval_required"})
        return "该工具涉及私有读取或写入操作，必须经过用户确认后才能执行。"
    for attempt in range(retries + 1):
        try:
            result = await asyncio.wait_for(tool.ainvoke(arguments), timeout=timeout_seconds)
            audit({"user_id": user_id, "thread_id": thread_id, "tool": tool.name, "risk": risk, "args_hash": argument_hash, "status": "success", "attempt": attempt + 1})
            return sanitize_output(result)
        except Exception as error:
            if attempt == retries:
                audit({"user_id": user_id, "thread_id": thread_id, "tool": tool.name, "risk": risk, "args_hash": argument_hash, "status": "failed", "error": type(error).__name__})
                return "工具调用失败或超时，已停止重试。"
    return "工具调用失败。"
