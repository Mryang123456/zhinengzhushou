"""注册内置工具及按用户绑定的记忆、Skill 和知识库工具。"""

import ast
import json
import math
import operator
from urllib.request import Request, urlopen

from ddgs import DDGS
from langchain_core.tools import ToolException, tool

from backend.app.knowledge.store import KnowledgeStore, pack_evidence
from backend.app.memory.store import MemoryStore
from backend.app.skills.registry import Skill
from backend.app.weather.service import WeatherQueryError, query_weather, weather_code_description


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
        result = evaluate_expression(expression)
        if not isinstance(result, (int, float)) or (isinstance(result, float) and not math.isfinite(result)):
            raise ToolException("计算结果超出支持的实数范围。")
        return str(result)
    except ZeroDivisionError as error:
        raise ToolException("计算失败：除数不能为零。") from error
    except (SyntaxError, ValueError, ArithmeticError) as error:
        raise ToolException("计算失败：表达式无效或超出支持的范围。") from error


def _get_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "langgraph-agent/1.0"})
    with urlopen(request, timeout=10) as response:  # noqa: S310 - 固定的 HTTPS API 地址
        return json.load(response)


@tool
def get_weather(location: str, date: str = "current", days: int = 1) -> str:
    """查询任意地点的当前天气或未来预报；date支持今天、明天、后天、星期或YYYY-MM-DD，days为1到7。"""
    try:
        return query_weather(location, date, days, _get_json)
    except WeatherQueryError as error:
        raise ToolException(str(error)) from error
    except (KeyError, OSError, TimeoutError, ValueError) as error:
        raise ToolException("天气服务暂未返回可用数据。") from error


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """检索公开网页中的最新信息，返回标题、摘要和链接。"""
    try:
        limit = max(1, min(max_results, 8))
        results = list(DDGS().text(query, max_results=limit))
    except Exception as error:
        raise ToolException("联网搜索失败，请稍后重试。") from error
    results = [item for item in results if str(item.get("href", "")).startswith(("http://", "https://"))]
    if not results:
        raise ToolException("联网搜索没有返回可引用的结果。")
    return json.dumps([
        {"title": item.get("title", ""), "body": item.get("body", ""), "href": item["href"]}
        for item in results
    ], ensure_ascii=False)


TOOLS = [calculator, get_weather, web_search]


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


def create_skill_tools(skills: list[Skill]):
    """提供可审计的 Skill 清单；Skill 指令只能由本地目录加载。"""
    @tool
    def list_skills() -> str:
        """查看当前智能体可用的专用技能及其用途。"""
        return "\n".join(f"- {skill.name}：{skill.description}" for skill in skills)

    return [list_skills]


def create_knowledge_tools(database_url: str | None, user_id: str):
    """仅在显式配置知识库数据库时暴露检索工具。"""
    if not database_url:
        return []
    store = KnowledgeStore(database_url)

    @tool
    def search_knowledge_base(query: str) -> str:
        """检索已导入的内部知识库，返回文本片段及其文档来源。"""
        evidence, _ = pack_evidence(store.search(query, user_id))
        if not evidence:
            raise ToolException("知识库中未找到可引用且你有权限访问的资料。")
        return evidence

    return [search_knowledge_base]
