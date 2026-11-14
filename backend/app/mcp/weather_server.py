"""将智能体的天气查询能力作为本地 MCP 工具提供。"""

from mcp.server.fastmcp import FastMCP

from backend.app.agent.tools import get_weather


mcp = FastMCP("weather")


@mcp.tool()
def current_weather(location: str, date: str = "current", days: int = 1) -> str:
    """查询任意地点的当前天气或未来1到7天预报。"""
    return get_weather.invoke({"location": location, "date": date, "days": days})


if __name__ == "__main__":
    mcp.run(transport="stdio")
