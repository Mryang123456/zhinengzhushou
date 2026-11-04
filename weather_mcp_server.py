"""将智能体的天气查询能力作为本地 MCP 工具提供。"""

from mcp.server.fastmcp import FastMCP

from agent import get_weather


mcp = FastMCP("weather")


@mcp.tool()
def current_weather(location: str) -> str:
    """查询城市或地点的当前天气。"""
    return get_weather.invoke({"location": location})


if __name__ == "__main__":
    mcp.run(transport="stdio")
