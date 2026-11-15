"""运行意图路由回归，使用合成对话和本地工具替身。"""

import argparse
import asyncio
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import tool

from backend.app.agent.service import build_agent
from backend.app.agent.turns import checkpoint_config, stream_events
from backend.app.config import MCP_CONFIG_FILE
from backend.app.intent.semantic import is_public_search_tool
from backend.app.mcp.governance import load_governance
from backend.app.models.factory import create_chat_model
from backend.app.weather.service import WeatherQueryError


class EvalModel:
    def __init__(self, case, live_router=None, live_plan=False):
        self.case = case
        self.live_router = live_router
        self.router_calls = 0
        self.router_ms = 0.0
        self.router_error = None
        self.live_plan = live_plan
        self.plan_calls = 0
        self.plan_ms = 0.0
        self.plan_error = None
        self.target_turn = False

    async def ainvoke(self, messages):
        request = json.loads(messages[-1].content)
        if not self.target_turn or "当前请求" not in request:
            return AIMessage(content="{}")
        self.router_calls += 1
        started = perf_counter()
        try:
            if self.live_router is not None:
                return await self.live_router.ainvoke(messages)
            fixture = self.case.get("model")
            return AIMessage(content=fixture if isinstance(fixture, str) else json.dumps(fixture or {}))
        except (Exception, asyncio.CancelledError) as error:
            self.router_error = "TimeoutError" if isinstance(error, asyncio.CancelledError) else type(error).__name__
            raise
        finally:
            self.router_ms += (perf_counter() - started) * 1000

    def bind_tools(self, _tools):
        return self

    async def astream(self, messages):
        prompt = messages[0].content
        last = messages[-1]
        if "你只负责拆解只读任务" in prompt:
            if self.target_turn and self.live_plan:
                self.plan_calls += 1
                started = perf_counter()
                try:
                    async for chunk in self.live_router.astream(messages):
                        yield chunk
                except (Exception, asyncio.CancelledError) as error:
                    self.plan_error = "TimeoutError" if isinstance(error, asyncio.CancelledError) else type(error).__name__
                    raise
                finally:
                    self.plan_ms += (perf_counter() - started) * 1000
            else:
                yield AIMessageChunk(content=json.dumps(self.case.get("plan", {"tasks": []})))
        elif "你是出行建议助手" in prompt:
            data = json.loads(last.content)
            yield AIMessageChunk(content=json.dumps({
                "activity": "建议调整行程。" if data["需要出游判断"] else "",
                "clothing": "建议带外套。" if data["需要穿衣建议"] else "",
            }, ensure_ascii=False))
        elif "上海现在适合穿什么衣服" == last.content:
            yield AIMessageChunk(content="", tool_calls=[{
                "name": "get_weather", "args": {"location": "上海"}, "id": "history-weather",
            }])
        elif self.target_turn and "call" in self.case and not isinstance(last, ToolMessage):
            yield AIMessageChunk(content="", tool_calls=[{**self.case["call"], "id": "fixture-call"}])
        elif isinstance(last, ToolMessage) and last.name == "calculator":
            yield AIMessageChunk(content="计算结果：" + last.content)
        else:
            yield AIMessageChunk(content="这是用于路由评估的示例回答。")


def fixture_weather(location, requested, conditions):
    condition = conditions.get(location, "小雨")
    if condition == "error":
        raise WeatherQueryError("模拟天气服务失败")
    offsets = {"今天": 0, "明天": 1, "后天": 2}
    if requested in offsets:
        day = date.today() + timedelta(days=offsets[requested])
        return f"{location}天气预报（{day}）：{condition}；最高气温 20°C。"
    return f"{location}当前天气：{condition}；气温 12°C。"


@tool
def search_knowledge_base(query: str) -> str:
    """检索合成的内部知识库。"""
    return "引用：knowledge://1#0\n来源：合成制度\n内容：报销需要提交凭证。\n【资料结束】"


async def evaluate_case(case, live_router, live_plan=False):
    model = EvalModel(case, live_router, live_plan)
    governance = load_governance(MCP_CONFIG_FILE)
    if "budget" in case:
        governance["maxToolCalls"] = case["budget"]
    config = checkpoint_config("eval")
    knowledge_enabled = case.get("knowledge", False)
    with (
        patch.dict("os.environ", {
            "KNOWLEDGE_DATABASE_URL": "fixture" if knowledge_enabled else "",
            "TOOL_RETRIEVAL_EMBEDDING_PROVIDER": "",
            "AGENT_DEEPSEEK_FETCH_ENABLED": "",
        }),
        patch("backend.app.agent.service.load_governance", return_value=governance),
        patch("backend.app.agent.service.create_chat_model", return_value=model),
        patch("backend.app.agent.service.create_knowledge_tools", return_value=[search_knowledge_base] if knowledge_enabled else []),
        patch("backend.app.agent.tools.query_weather", side_effect=lambda location, requested, *_:
              fixture_weather(location, requested, case.get("weather", {}))),
        patch("backend.app.agent.tools.DDGS") as search,
    ):
        search.return_value.text.return_value = [{"href": "https://example.org/report", "body": "这是一条合成评估资料。"}]
        if case.get("search_error"):
            search.return_value.text.side_effect = OSError("模拟搜索失败")
        graph = await build_agent(memory_db_path=None)
        history = case.get("history", [])
        for question in [history] if isinstance(history, str) else history:
            async for _ in stream_events(graph, question, "eval"):
                pass
        history_size = len((await graph.aget_state(config)).values.get("messages", []))
        model.target_turn = True
        events = [event async for event in stream_events(graph, case["question"], "eval")]
        state = (await graph.aget_state(config)).values
    messages = state["messages"][history_size:]
    attempted = {name for event in events if event["type"] == "tool_start" for name in event.get("tools", [])}
    tools = {message.name for message in messages if isinstance(message, ToolMessage) and message.status == "success"}
    required = set(case.get("require_tools", []))
    forbidden = set(case.get("forbid_tools", []))
    if case["expected"].get("status") == "clarify":
        forbidden.update(attempted | tools)
    if case["expected"].get("no_public_search"):
        forbidden.update(name for name in attempted | tools if is_public_search_tool(name))
    mismatches = [key for key, expected in case["expected"].items() if state["intent"].get(key) != expected]
    resolved = state["intent"].get("resolved_question", "")
    if not all(part in resolved for part in case.get("resolved_contains", [])) or any(
        part in resolved for part in case.get("resolved_excludes", [])
    ):
        mismatches.append("resolved_question")
    ledger = Counter(name for event in events if event["type"] == "tool_start" for name in event.get("tools", []))
    if "tool_counts" in case and ledger != case["tool_counts"]:
        mismatches.append("tool_counts")
    answer = events[-1].get("content", "")
    if not all(part in answer for part in case.get("answer_contains", [])):
        mismatches.append("answer")
    if "expected_tasks" in case:
        tasks = state["intent"].get("tasks", [])
        if len(tasks) != len(case["expected_tasks"]) or any(
            any(task.get(key) != value for key, value in expected.items())
            for task, expected in zip(tasks, case["expected_tasks"])
        ):
            mismatches.append("tasks")
    return {
        "id": case["id"],
        "route_match": not mismatches,
        "mismatched_fields": mismatches,
        "missing_tools": sorted(required - tools),
        "unexpected_tools": sorted(forbidden & (attempted | tools)),
        "completed": events[-1]["type"] == "done",
        "router_calls": model.router_calls,
        "router_error": model.router_error,
        "router_ms": round(model.router_ms, 2),
        "plan_calls": model.plan_calls,
        "plan_ms": round(model.plan_ms, 2),
        "plan_error": model.plan_error,
    }


async def run(cases, mode, live_plan=False):
    live_router = create_chat_model() if mode == "live" else None
    skipped = [case["id"] for case in cases if mode == "live" and case.get("fixture_only")]
    cases = [case for case in cases if case["id"] not in skipped]
    if not cases:
        raise ValueError("当前模式没有可评估的用例")
    results = [await evaluate_case(case, live_router, live_plan) for case in cases]
    total = len(results)
    timings = [item["router_ms"] for item in results if item["router_calls"]]
    failures = [item["id"] for item in results if not item["route_match"] or not item["completed"]
                or item["missing_tools"] or item["unexpected_tools"] or item["router_error"] or item["plan_error"]]
    plan_timings = [item["plan_ms"] for item in results if item["plan_calls"]]
    return {
        "mode": mode, "cases": total,
        "route_match_rate": sum(item["route_match"] for item in results) / total,
        "unexpected_tool_case_rate": sum(bool(item["unexpected_tools"]) for item in results) / total,
        "router_calls": sum(item["router_calls"] for item in results),
        "router_error_cases": sum(bool(item["router_error"]) for item in results),
        "skipped_cases": skipped,
        "router_median_ms": median(timings) if timings else None,
        "planner_calls": sum(item["plan_calls"] for item in results),
        "planner_median_ms": median(plan_timings) if plan_timings else None,
        "failed_cases": failures, "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--mode", choices=("fixture", "live"), default="fixture")
    parser.add_argument("--live-plan", action="store_true", help="同时调用真实模型拆解复合任务")
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        parser.error("评估集不能为空，且用例编号必须唯一")
    if args.live_plan and args.mode != "live":
        parser.error("--live-plan 需要 --mode live")
    report = asyncio.run(run(cases, args.mode, args.live_plan))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(report["failed_cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
