"""生成联网正文所需的编号资料和写作约束。"""

import json
import re
from urllib.parse import urlparse

from langchain_core.messages import BaseMessage, ToolMessage

from backend.app.content_safety import contains_secret, is_direct_prompt_override
from backend.app.models.tool_protocol import contains_textual_tool_protocol
from backend.app.prompts import prompt_text
from backend.app.quality.validator import URL_PATTERN, web_result_records

def web_writing_rules() -> str:
    return prompt_text("web.writing")


def numbered_web_excerpts(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "编号": f"S{index}", "网站": urlparse(source["url"]).hostname or "",
            "标题": source["title"], "日期": source["date"],
            "类型": source["kind"], "摘录": source["excerpt"],
        }
        for index, source in enumerate(sources, 1)
    ]


def web_writing_context(sources: list[dict[str, str]]) -> str:
    return web_writing_rules() + "\n本轮编号资料：\n" + json.dumps(numbered_web_excerpts(sources), ensure_ascii=False)


def render_evidence_table(table: dict | None, source_count: int) -> str:
    if not table:
        return ""

    def clean(value: str) -> str:
        text = " ".join(value.split())[:240]
        return text.translate(str.maketrans("|<>[]\x60\\", "｜＜＞［］｀＼"))

    columns = [clean(value) for value in table["columns"]]
    if not all(columns):
        return ""
    rows = []
    for row in table["rows"]:
        if len(row["cells"]) != len(columns) or not all(1 <= source <= source_count for source in row["sources"]):
            continue
        cells = [clean(value) or "—" for value in row["cells"]]
        cells[-1] += " " + "".join(f"[S{source}]" for source in dict.fromkeys(row["sources"]))
        rows.append("| " + " | ".join(cells) + " |")
    if len(set(rows)) < 2:
        return ""
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *dict.fromkeys(rows),
    ])


def include_evidence_table(answer: str, table: str, title: str = "") -> str:
    if not table or not answer.strip() or re.search(r"(?m)^\s*\|?\s*:?-{3,}:?\s*\|\s*:?-{3,}", answer):
        return answer
    first, separator, rest = answer.strip().partition("\n\n")
    title = " ".join(title.split()).translate(str.maketrans("<>[]\x60\\", "＜＞［］｀＼")).strip("# ") or "明细一览"
    return first + f"\n\n### {title}\n\n" + table + (separator + rest if rest else "")


def render_evidence_details(details: list[dict], sources: list[dict[str, str]], *, answer: str = "") -> str:
    rows = {}
    for detail in details:
        label = " ".join(detail["label"].split())
        fact = " ".join(detail["text"].split())
        identifiers = [
            index for index in dict.fromkeys(detail["sources"])
            if 1 <= index <= len(sources) and sources[index - 1]["kind"] != "仅标题"
        ]
        if not label or not fact or fact in answer or not identifiers:
            continue
        if any(check(label + " " + fact) for check in (
            contains_secret, is_direct_prompt_override, contains_textual_tool_protocol, URL_PATTERN.search,
        )):
            continue
        translation = str.maketrans("*[]<>`\\", "＊［］＜＞｀＼")
        label, fact = label.translate(translation), fact.translate(translation)
        rows.setdefault(fact, f"- **{label}**：{fact} " + "".join(f"[S{index}]" for index in identifiers))
    return "\n".join(rows.values())


def web_writing_turn(turn: list[BaseMessage], sources: list[dict[str, str]]) -> list[BaseMessage]:
    """把模型可见的网页返回替换为所选摘录，并保留工具调用配对。"""
    pending = dict(zip((source["url"] for source in sources), numbered_web_excerpts(sources)))
    prepared: list[BaseMessage] = []
    for message in reversed(turn):
        if isinstance(message, ToolMessage) and message.status == "success" and isinstance(message.content, str):
            items = []
            for record in web_result_records(message.content):
                excerpt = pending.pop(record["href"], None)
                if excerpt is not None:
                    items.append(excerpt)
            message = message.model_copy(update={"content": json.dumps(items, ensure_ascii=False)})
        prepared.append(message)
    return prepared[::-1]
