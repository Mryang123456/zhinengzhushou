"""基于 LangChain 与 LangGraph 的基础工具型智能体，使用内存检查点保存会话。"""

from __future__ import annotations

import argparse
import asyncio
import ast
import json
import operator
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from memory import MemoryStore, create_memory_store
from models import ModelConfigurationError, create_chat_model, ensure_model_ready, get_model_settings

load_dotenv(Path(__file__).with_name(".env.local"))

SYSTEM_PROMPT = """你是一名简洁、可靠的中文助手。
你可以在任意回合调用工具，也可以在回复前连续调用多个工具。
计算时使用 calculator；查询当前天气时使用 get_weather；查询最新或可由外部验证的信息时使用 web_search。
回答中请附上网页检索结果的链接。除非本回合调用了相关工具，否则不要声称信息是最新的。
工具失败时，简要说明限制，不要编造答案。"""

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def evaluate_expression(expression: str) -> int | float:
    """计算基础算术表达式；禁止名称、函数调用和属性访问。"""

    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](visit(node.operand))
        raise ValueError("只允许数字、+、-、*、/、//、%、** 和括号。")

    if len(expression) > 200:
        raise ValueError("表达式过长。")
    return visit(ast.parse(expression, mode="eval"))


@tool
def calculator(expression: str) -> str:
    """计算基础算术表达式，例如“(17 + 3) * 2”。"""
    try:
        return str(evaluate_expression(expression))
    except (SyntaxError, ValueError, ZeroDivisionError) as error:
        return f"计算失败：{error}"


def _get_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "langgraph-agent/1.0"})
    with urlopen(request, timeout=10) as response:  # noqa: S310 - 固定的 HTTPS API 地址
        return json.load(response)


def weather_code_description(code: int) -> str:
    descriptions = {
        0: "晴朗", 1: "大致晴朗", 2: "局部多云", 3: "阴天", 45: "有雾", 48: "雾凇",
        51: "毛毛雨", 53: "毛毛雨", 55: "强毛毛雨", 61: "小雨", 63: "中雨",
        65: "大雨", 71: "小雪", 73: "中雪", 75: "大雪", 80: "阵雨",
        81: "中等阵雨", 82: "强阵雨", 95: "雷暴", 96: "伴冰雹的雷暴", 99: "强冰雹雷暴",
    }
    return descriptions.get(code, f"天气代码 {code}")


@tool
def get_weather(location: str) -> str:
    """查询城市或地点的当前天气，例如“新加坡”或“北京”。"""
    try:
        geocode_url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(
            {"name": location, "count": 1, "language": "zh", "format": "json"}
        )
        matches = _get_json(geocode_url).get("results", [])
        if not matches:
            return f"未找到地点：{location}"
        place = matches[0]
        forecast_url = "https://api.open-meteo.com/v1/forecast?" + urlencode(
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                "timezone": "auto",
            }
        )
        current = _get_json(forecast_url)["current"]
        return (
            f"{place['name']}当前天气：{weather_code_description(current['weather_code'])}；"
            f"气温 {current['temperature_2m']}°C，体感 {current['apparent_temperature']}°C，"
            f"湿度 {current['relative_humidity_2m']}%，风速 {current['wind_speed_10m']} km/h，"
            f"降水 {current['precipitation']} mm。"
        )
    except (KeyError, OSError, TimeoutError, ValueError) as error:
        return f"天气查询失败：{error}"


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """检索公开网页中的最新信息，返回标题、摘要和链接。"""
    try:
        limit = max(1, min(max_results, 8))
        results = list(DDGS().text(query, max_results=limit))
        if not results:
            return f"未找到与“{query}”相关的搜索结果。"
        return "\n".join(
            f"{index}. {item.get('title', '无标题')}\n{item.get('body', '')}\n{item.get('href', '')}"
            for index, item in enumerate(results, start=1)
        )
    except Exception as error:
        return f"联网搜索失败：{error}"


TOOLS = [calculator, get_weather, web_search]


def content_to_text(content: object) -> str:
    """将模型的文本或内容块统一转换为可显示文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content or "")


def is_billing_error(error: Exception) -> bool:
    """识别 OpenAI 的常见计费或额度错误文案。"""
    message = str(error).lower()
    return any(
        phrase in message
        for phrase in (
            "account is not active",
            "no credits remaining",
            "insufficient_quota",
            "credit_balance_exhausted",
        )
    )


def format_memory_context(memories: list[tuple[int, str]]) -> str:
    if not memories:
        return "（暂无已保存的长期记忆。）"
    return "\n".join(f"- [{memory_id}] {content}" for memory_id, content in memories)


def create_memory_tools(memory_store: MemoryStore, user_id: str):
    """为当前用户创建受隔离的长期记忆工具。"""

    @tool
    def remember_memory(content: str) -> str:
        """仅当用户明确要求“记住”某条非敏感信息时，保存为长期记忆。"""
        try:
            memory_id = memory_store.remember(user_id, content)
            return f"已保存长期记忆，编号为 {memory_id}。"
        except ValueError as error:
            return f"无法保存记忆：{error}"

    @tool
    def recall_memory(query: str) -> str:
        """检索当前用户已保存的长期记忆。"""
        memories = memory_store.search(user_id, query)
        return format_memory_context(memories)

    @tool
    def forget_memory(memory_id: int) -> str:
        """仅当用户明确要求删除时，删除指定编号的一条长期记忆。"""
        if memory_store.forget(user_id, memory_id):
            return f"已删除编号为 {memory_id} 的长期记忆。"
        return f"未找到编号为 {memory_id} 的长期记忆，或该记忆不属于当前用户。"

    return [remember_memory, recall_memory, forget_memory]


def load_mcp_connections(config_path: Path) -> dict:
    """加载显式 MCP 服务配置；绝不从聊天输入中接受服务启动命令。"""
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        connections = config["mcpServers"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"MCP 配置无效：{config_path}；原因：{error}") from error
    if not isinstance(connections, dict) or not connections:
        raise ValueError("mcpServers 必须是非空对象。")
    return connections


@asynccontextmanager
async def checkpoint_scope(database_url: str | None = None):
    """提供检查点存储；配置 PostgreSQL 时，任务状态可跨进程恢复。"""
    if not database_url:
        yield MemorySaver()
        return

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool
    except ImportError as error:
        raise RuntimeError(
            "缺少 PostgreSQL 检查点依赖，请重新安装 requirements.txt。"
        ) from error

    async with AsyncConnectionPool(
        conninfo=database_url,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        yield checkpointer


async def build_agent(
    mcp_config_path: Path | None = None,
    user_id: str = "default",
    memory_db_path: Path | None = Path(__file__).with_name("data").joinpath("memory.db"),
    memory_database_url: str | None = None,
    checkpointer=None,
):
    """构建同时包含本地工具和已发现 MCP 工具的工作流。"""
    tools = TOOLS.copy()
    memory_store = create_memory_store(memory_database_url, memory_db_path)
    if memory_store:
        tools.extend(create_memory_tools(memory_store, user_id))
    if mcp_config_path:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(
            load_mcp_connections(mcp_config_path), tool_name_prefix=True
        )
        tools.extend(await client.get_tools())

    model = create_chat_model().bind_tools(tools)

    async def call_model(state: MessagesState):
        writer = get_stream_writer()
        response = None
        memory_context = (
            format_memory_context(memory_store.recent(user_id))
            if memory_store
            else "（长期记忆功能未启用。）"
        )
        memory_instruction = (
            "\n\n当前用户的长期记忆如下，仅在与问题相关时使用：\n"
            f"{memory_context}\n"
            "只有用户明确要求记住时才能调用 remember_memory；"
            "只有用户明确要求删除时才能调用 forget_memory。"
        )
        async for chunk in model.astream(
            [SystemMessage(SYSTEM_PROMPT + memory_instruction), *state["messages"]]
        ):
            text = content_to_text(chunk.content)
            if text:
                writer({"type": "token", "content": text})
            response = chunk if response is None else response + chunk
        if response is None:
            raise RuntimeError("模型未返回任何内容。")
        if response.tool_calls:
            writer(
                {
                    "type": "tool_start",
                    "tools": [tool_call["name"] for tool_call in response.tool_calls],
                }
            )
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer or MemorySaver())


def checkpoint_config(thread_id: str, user_id: str = "default") -> dict:
    """生成按用户隔离的检查点键，防止不同用户共享同一会话状态。"""
    return {"configurable": {"thread_id": f"{user_id}:{thread_id}"}, "recursion_limit": 12}


async def ask(app, question: str, thread_id: str, user_id: str = "default") -> str:
    """执行一个回合；重用 thread_id 可在当前进程内保持对话。"""
    result = await app.ainvoke(
        {"messages": [HumanMessage(question)]},
        config=checkpoint_config(thread_id, user_id),
    )
    return result["messages"][-1].content


async def stream_events(app, question: str, thread_id: str, user_id: str = "default"):
    """逐步产出 token、工具状态和最终答案，供命令行或 SSE 接口复用。"""
    config = checkpoint_config(thread_id, user_id)
    final_answer = ""
    async for mode, payload in app.astream(
        {"messages": [HumanMessage(question)]},
        config=config,
        stream_mode=["custom", "updates"],
    ):
        if mode == "custom":
            yield payload
            continue
        for node_name, update in payload.items():
            messages = update.get("messages", [])
            if node_name == "tools":
                tool_names = [getattr(message, "name", "工具") for message in messages]
                yield {"type": "tool_end", "tools": tool_names}
            elif node_name == "agent" and messages:
                candidate = content_to_text(messages[-1].content)
                if candidate:
                    final_answer = candidate
    yield {"type": "done", "content": final_answer}


async def answer_or_exit(app, question: str, thread_id: str, user_id: str = "default") -> str:
    """将常见账户配置失败转换为可执行的命令行提示。"""
    try:
        return await ask(app, question, thread_id, user_id)
    except Exception as error:
        if is_billing_error(error):
            raise SystemExit(
                "OpenAI API 项目尚未启用。请先在 OpenAI Platform 启用 API 计费后重试。"
            ) from error
        raise


async def run_cli(args: argparse.Namespace) -> None:
    mcp_config_path = None if args.no_mcp else Path(args.mcp_config)
    memory_db_path = None if args.no_memory else Path(args.memory_db)
    checkpoint_database_url = args.checkpoint_database_url or os.getenv("CHECKPOINT_DATABASE_URL")
    async with checkpoint_scope(checkpoint_database_url) as checkpointer:
        app = await build_agent(
            mcp_config_path,
            args.user_id,
            memory_db_path,
            args.memory_database_url or os.getenv("MEMORY_DATABASE_URL"),
            checkpointer,
        )
        if args.once:
            if args.no_stream:
                print(await answer_or_exit(app, args.once, args.thread_id, args.user_id))
            else:
                await print_stream(app, args.once, args.thread_id, args.user_id)
            return

        print("智能体已就绪。输入 /quit 退出。")
        while True:
            question = input("> ").strip()
            if question in {"/quit", "/exit"}:
                return
            if question:
                if args.no_stream:
                    print(await answer_or_exit(app, question, args.thread_id, args.user_id))
                else:
                    await print_stream(app, question, args.thread_id, args.user_id)


async def print_stream(app, question: str, thread_id: str, user_id: str = "default") -> None:
    """将流式事件渲染为中文命令行输出。"""
    printed_token = False
    try:
        async for event in stream_events(app, question, thread_id, user_id):
            if event["type"] == "token":
                print(event["content"], end="", flush=True)
                printed_token = True
            elif event["type"] == "tool_start":
                print(f"\n[正在调用工具：{'、'.join(event['tools'])}]", flush=True)
            elif event["type"] == "tool_end":
                print(f"\n[工具调用完成：{'、'.join(event['tools'])}]", flush=True)
            elif event["type"] == "done":
                if not printed_token:
                    print(event["content"], end="", flush=True)
                print()
    except Exception as error:
        if is_billing_error(error):
            raise SystemExit(
                "OpenAI API 项目尚未启用。请先在 OpenAI Platform 启用 API 计费后重试。"
            ) from error
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="基于 LangChain 和 LangGraph 的基础智能体")
    parser.add_argument("--once", help="执行一个问题后退出")
    parser.add_argument("--thread-id", default="demo", help="对话检查点标识")
    parser.add_argument(
        "--mcp-config",
        default=str(Path(__file__).with_name("mcp.json")),
        help="显式 MCP 服务配置文件的路径",
    )
    parser.add_argument("--no-mcp", action="store_true", help="仅启用内置工具")
    parser.add_argument("--no-stream", action="store_true", help="关闭命令行流式输出")
    parser.add_argument("--user-id", default="default", help="长期记忆所属用户标识")
    parser.add_argument(
        "--memory-db",
        default=str(Path(__file__).with_name("data").joinpath("memory.db")),
        help="SQLite 长期记忆数据库路径",
    )
    parser.add_argument("--no-memory", action="store_true", help="关闭长期记忆功能")
    parser.add_argument(
        "--memory-database-url",
        help="PostgreSQL 长期记忆数据库地址；优先级高于本地 SQLite",
    )
    parser.add_argument(
        "--checkpoint-database-url",
        help="PostgreSQL 检查点数据库地址；用于跨重启恢复任务状态",
    )
    args = parser.parse_args()

    try:
        ensure_model_ready(get_model_settings())
    except ModelConfigurationError as error:
        raise SystemExit(f"模型配置错误：{error}") from error

    asyncio.run(run_cli(args))


if __name__ == "__main__":
    main()
