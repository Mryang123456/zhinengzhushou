"""发现服务端配置中获准使用的 MCP 工具。"""

import asyncio
import logging
from pathlib import Path

from backend.app.mcp.governance import is_registered_tool, load_mcp_connections, search_tool_with_policy_schema

logger = logging.getLogger(__name__)


async def discover_mcp_tools(config_path: Path, governance: dict) -> list:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    connections = load_mcp_connections(config_path)
    client = MultiServerMCPClient(connections, tool_name_prefix=True)
    try:
        discovered = await asyncio.wait_for(client.get_tools(), timeout=5)
    except Exception as error:
        logger.warning("MCP 工具发现失败，继续使用本地工具：%s", type(error).__name__)
        return []
    return [search_tool_with_policy_schema(tool, governance) for tool in discovered
            if is_registered_tool(tool.name, governance)]
