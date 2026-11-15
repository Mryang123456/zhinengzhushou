"""在一个回合内控制公开网页去重和证据预算。"""

from collections import deque

from langchain_core.messages import ToolMessage

from backend.app.intent.semantic import is_public_search_tool
from backend.app.quality.validator import web_result_records, web_result_urls

WEB_RESULT_LIMIT = 50
WEB_RETRY_RESERVE = 5


def select_web_results(previous_turn, batch: list[ToolMessage], *, reserve_retry: bool) -> dict[str, list[dict]]:
    seen = web_result_urls(previous_turn) or set()
    prior_bodies = {
        record["href"]: record["body"]
        for message in previous_turn if isinstance(message, ToolMessage) and message.status == "success"
        and isinstance(message.content, str) and message.name and is_public_search_tool(message.name)
        for record in web_result_records(message.content) if record["href"] in seen
    }
    limit = WEB_RESULT_LIMIT - WEB_RETRY_RESERVE if reserve_retry else WEB_RESULT_LIMIT
    selected: dict[str, list[dict]] = {message.tool_call_id: [] for message in batch}
    pending = deque((message.tool_call_id, iter(web_result_records(message.content))) for message in batch)
    refreshed = set()
    while pending and len(seen) < limit:
        call_id, records = pending.popleft()
        for record in records:
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
