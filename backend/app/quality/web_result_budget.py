"""在一个回合内控制公开网页去重和证据预算。"""

from collections import deque

from langchain_core.messages import ToolMessage

from backend.app.intent.semantic import is_public_search_tool
from backend.app.quality.validator import web_result_records, web_result_urls

WEB_RESULT_LIMIT = 50
WEB_RETRY_RESERVE = 5


def select_web_results(
    previous_turn, batch: list[ToolMessage], *, reserve_retry: bool, reserved_pages: int = WEB_RETRY_RESERVE,
    per_tool_limit: int | None = None,
) -> dict[str, list[dict]]:
    seen = web_result_urls(previous_turn) or set()
    prior_bodies = {
        record["href"]: record["body"]
        for message in previous_turn if isinstance(message, ToolMessage) and message.status == "success"
        and isinstance(message.content, str) and message.name and is_public_search_tool(message.name)
        for record in web_result_records(message.content) if record["href"] in seen
    }
    limit = WEB_RESULT_LIMIT - reserved_pages if reserve_retry else WEB_RESULT_LIMIT
    selected: dict[str, list[dict]] = {message.tool_call_id: [] for message in batch}
    admitted: dict[str, set[str]] = {message.tool_call_id: set() for message in batch}
    pending = deque((message.tool_call_id, iter(web_result_records(message.content))) for message in batch)
    refreshed = set()
    while pending and (len(seen) < limit or per_tool_limit is not None):
        call_id, records = pending.popleft()
        for record in records:
            if per_tool_limit is not None:
                # 各子任务独立使用同一网页，整轮网页额度仍按唯一链接计算。
                url = record["href"]
                if url in admitted[call_id] or (url not in seen and len(seen) >= limit):
                    continue
                admitted[call_id].add(url)
                seen.add(url)
                selected[call_id].append(record)
                if len(selected[call_id]) < per_tool_limit:
                    pending.append((call_id, records))
                break
            if record["href"] in seen:
                url = record["href"]
                if url in prior_bodies and url not in refreshed and record["body"] and record["body"] != prior_bodies[url]:
                    selected[call_id].append(record)
                    refreshed.add(url)
                continue
            seen.add(record["href"])
            selected[call_id].append(record)
            pending.append((call_id, records))
            break
    return selected
