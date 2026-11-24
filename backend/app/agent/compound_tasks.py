"""执行复合只读任务，并逐项核对工具结果。"""

import asyncio
import json
import logging
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer

from backend.app.agent.context import message_text
from backend.app.agent.evidence_retry import parse_evidence_decision, retry_prompt, validated_query
from backend.app.intent.planning import forecast_has_rain
from backend.app.models.tool_protocol import contains_textual_tool_protocol
from backend.app.observability import event, observe_duration
from backend.app.prompts import prompt_text
from backend.app.quality.recommendations import (
    Recommendations, enrich_recommendation_sources, recommendation_constraints, verified_recommendations,
)
from backend.app.quality.validator import render_web_citations, validate_answer, weather_tool_answer, web_answer_sources
from backend.app.quality.web_writing import web_writing_context

logger = logging.getLogger(__name__)
PLAN_WEB_RESULT_LIMIT = 5


def task_location(tasks, index):
    task = tasks[index]
    dependency = task["depends_on"]
    return task["location"] or (tasks[dependency]["location"] if dependency is not None else "")


def task_search_query(tasks, index):
    task = tasks[index]
    topic = re.sub(r"^(?:请|帮我)?(?:推荐|查找|查询|搜索|找)(?:[一二两三四五六七八九十\d]+个)?", "", task["text"]).strip()
    return f"{task_location(tasks, index)} {topic or task['text']}".strip()


def planned_weather(question, task, result):
    if not result:
        return ""
    lookup = f"{task['location']} {task['date']} 天气"
    report = weather_tool_answer(lookup, [result]) if task["location"] in question else ""
    return report if report and validate_answer(lookup, report, [result])["ok"] else ""


def plan_dependency_status(question, tasks, index, plan_id, results):
    task = tasks[index]
    dependency = task["depends_on"]
    if dependency is None:
        return "ready"
    result = results.get(f"{plan_id}-{dependency}")
    if result is None:
        return "waiting"
    if result.status != "success":
        return "blocked"
    if task["condition"] != "if_rain":
        return "ready"
    rain = forecast_has_rain(planned_weather(question, tasks[dependency], result))
    return {True: "ready", False: "skipped", None: "blocked"}[rain]


def planned_tool_results(messages):
    return {message.tool_call_id: message for message in messages if isinstance(message, ToolMessage)}


def plan_tool_call(name, call_id, args):
    get_stream_writer()({"type": "tool_start", "tools": [name]})
    return {"messages": [AIMessage(content="", tool_calls=[{"name": name, "id": call_id, "args": args}])],
            "allowed_tool_names": [name]}


async def review_plan_search(tasks, plan_id, messages, chat_model, *, used, max_batch, max_total):
    """复核刚完成的公开搜索。补查只属于当前子任务，最多一次。"""
    last = messages[-1]
    if not max_batch or not isinstance(last, ToolMessage) or last.name != "web_search":
        return {}
    results = planned_tool_results(messages)
    pending = sum(f"{plan_id}-{index}" not in results for index in range(len(tasks)))
    if used + pending >= max_total:
        return {}
    index = next((index for index in range(len(tasks)) if last.tool_call_id == f"{plan_id}-{index}"), None)
    if index is None or f"{last.tool_call_id}-retry" in results:
        return {}
    invoke = getattr(chat_model(), "ainvoke", None)
    if not callable(invoke):
        return {}

    task = tasks[index]
    question = f"{task_location(tasks, index)} {task['text']}".strip()
    scoped = [message for message in messages if isinstance(message, AIMessage)
              and any(call["id"] == last.tool_call_id for call in message.tool_calls)]
    prompt = [
        SystemMessage(prompt_text("compound.review")),
        HumanMessage(retry_prompt(question, "检查当前子任务的原始要求是否已满足", [*scoped, last], knowledge=False)),
    ]
    started = time.monotonic()
    try:
        async with asyncio.timeout(6):
            response = await invoke(prompt)
        decision = parse_evidence_decision(message_text(response), messages)
        if decision.action != "search":
            return {}
        query = validated_query(json.dumps({"query": f"{task_search_query(tasks, index)} {decision.query}"}), messages)
    except Exception as error:
        event("plan_search_review_fallback", reason=type(error).__name__, step=index)
        return {}
    finally:
        observe_duration("plan_search_review", time.monotonic() - started)
    if not query:
        return {}
    event("plan_search_supplement", step=index)
    get_stream_writer()({"type": "evidence_retry", "reason": "复合任务的子项缺少资料", "tool": "web_search"})
    return plan_tool_call("web_search", f"{last.tool_call_id}-retry", {"query": query})


def run_plan_step(question, tasks, plan_id, messages, used, *, max_batch, max_total):
    results = planned_tool_results(messages)
    for index, task in enumerate(tasks):
        call_id = f"{plan_id}-{index}"
        if call_id in results:
            continue
        location = task_location(tasks, index)
        if plan_dependency_status(question, tasks, index, plan_id, results) != "ready":
            continue
        if not max_batch or used >= max_total:
            break
        if task["kind"] == "weather":
            name = "get_weather"
            args = {"location": location, "date": task["date"] or "current"}
        else:
            name = "web_search"
            args = {"query": task_search_query(tasks, index)}
        return plan_tool_call(name, call_id, args)
    return {"draft_answer": ""}


async def compose_plan_answer(question, plan, turn_messages, chat_model):
    plan_id = plan["plan_id"]
    results = planned_tool_results(turn_messages)
    sections = []
    outcomes = {}
    for index, task in enumerate(plan["tasks"]):
        result = results.get(f"{plan_id}-{index}")
        if task["kind"] == "weather":
            report = planned_weather(question, task, result)
            sections.append(report or f"{task['location']}（{task['date'] or '当前'}）未取得匹配的天气资料。")
            outcomes[index] = "complete" if report else "blocked"
            continue
        if not result and task["condition"] == "if_rain":
            status = plan_dependency_status(question, plan["tasks"], index, plan_id, results)
            if status != "ready":
                sections.append("预报未显示降雨，因此没有执行条件推荐。" if status == "skipped"
                                else "暂时无法确认降雨条件，因此没有执行条件推荐。")
                outcomes[index] = "skipped" if status == "skipped" else "blocked"
                continue
        search_results = [message for message in (result, results.get(f"{plan_id}-{index}-retry")) if message]
        location = task_location(plan["tasks"], index)
        sources = web_answer_sources(f"{location} {task['text']}".strip(), search_results)
        indoor, family = recommendation_constraints(task["text"])
        if indoor or family:
            count = task["count"] or 3
            lines = []
            if sources:
                get_stream_writer()({"type": "progress", "stage": "verifying_recommendations"})
                sources = await enrich_recommendation_sources(sources, location, indoor=indoor, family=family)
                prompt = SystemMessage(
                    prompt_text("recommendation.extract")
                    + "\n"
                    + json.dumps(sources, ensure_ascii=False)
                )
                try:
                    async with asyncio.timeout(12):
                        request = HumanMessage(f"{task['text']}\n地点：{location}\n最多 {count} 项")
                        parts = [message_text(chunk) async for chunk in chat_model().astream([prompt, request])]
                    extracted = Recommendations.model_validate_json("".join(parts))
                    lines = verified_recommendations(extracted, sources, location, indoor, family, count)
                except Exception as error:
                    logger.info("景点证据提取未完成：%s", type(error).__name__)
            outcomes[index] = "complete" if len(lines) >= count else "partial" if lines else "blocked"
            if len(lines) < count:
                lines.append("资料不足，暂未核实更多同时满足这些条件的景点。" if lines
                             else "本轮没有核实到同时满足这些条件的景点。")
            sections.append("\n".join(lines))
            continue
        section = ""
        if sources:
            prompt = SystemMessage(
                web_writing_context(sources)
                + "\n"
                + prompt_text("compound.web_answer")
            )
            try:
                async with asyncio.timeout(12):
                    request_text = task["text"] + (f"\n地点：{location}" if location else "")
                    if task["count"]:
                        request_text += f"\n最多给出 {task['count']} 项，资料不足时可少给。"
                    request = HumanMessage(request_text)
                    parts = [message_text(chunk) async for chunk in chat_model().astream([prompt, request])]
                draft = render_web_citations("".join(parts), sources)
                if draft and not contains_textual_tool_protocol(draft) and validate_answer(
                    task["text"], draft, search_results, requires_web_evidence=True,
                )["ok"]:
                    section = draft
            except Exception as error:
                logger.info("子任务正文整理未完成：%s", type(error).__name__)
        sections.append(section or "这一项暂未取得可核验的回答。")
        outcomes[index] = "complete" if section else "blocked"
    statuses = list(outcomes.values())
    unfinished = any(status in {"partial", "blocked"} for status in statuses)
    partial = unfinished and any(status in {"complete", "partial"} for status in statuses)
    event(
        "plan_completed", complete=statuses.count("complete"), partial=statuses.count("partial"),
        skipped=statuses.count("skipped"), blocked=statuses.count("blocked"),
    )
    get_stream_writer()({
        "type": "quality", "passed": not unfinished or partial, "partial": partial,
        "reason": "已逐项校验复合任务",
    })
    return "\n\n".join(sections), outcomes
