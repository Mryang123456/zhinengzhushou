"""校验回答的来源、数字及需要逐句核对的证据。"""

import json
import re
from datetime import date as Date, timedelta
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, ToolMessage

from backend.app.content_safety import contains_secret
from backend.app.intent.policy import requested_knowledge_query
from backend.app.quality.numeric_evidence import supports_numeric_claim
from backend.app.quality.web_passages import extract_web_passages

FRESHNESS_TERMS = ("最新", "今天", "今日", "目前", "当前", "实时", "最近", "现行")
INTERNAL_FRESHNESS_TERMS = ("最新", "今天", "当前", "实时", "现行", "最近")
STRICT_WEB_TERMS = ("政策", "法律", "价格")
HIGH_RISK_TERMS = ("诊断", "用药", "疾病", "法律", "起诉", "合同", "投资", "股票", "理财", "贷款")
URL_PATTERN = re.compile(r"""https?://[^\s<>()\[\]{}"'，。！？；、]+""")
KNOWLEDGE_PATTERN = re.compile(r"knowledge://\d+#\d+(?!\d)")
REFERENCE_PATTERN = re.compile(f"{URL_PATTERN.pattern}|{KNOWLEDGE_PATTERN.pattern}")
MARKDOWN_WEB_LINK = re.compile(rf"(?<!!)\[([^\]\n]*)\]\(({URL_PATTERN.pattern})\)")
NUMBERED_WEB_LINK = re.compile(rf"(?<!!)\[来源 \d+\]\(({URL_PATTERN.pattern})\)")
WEB_SOURCE_ID = re.compile(r"\[S([1-9]\d*)\]")
WEB_LIST_ITEM = re.compile(r"^(?:[-*+•]|\d+[.)、]|[一二三四五六七八九十]+、)\s*")
WEB_SOURCE_LABEL = re.compile(r"(?:参考(?:资料|文献)?|消息来源|新闻来源|参考来源|来源链接|来源|链接|出处|报道|sources?|references?)[：:]?", re.I)
WEB_SECTION_LABEL = re.compile(r"(?:以下是)?(?:近期|最近|最新)?(?:新闻(?:摘要|动态|速览|要点)?|动态|简报|摘要|主要动态|主要消息|消息汇总|相关报道)(?:如下)?[：:]?")
WEB_LAYOUT_HEADING = re.compile(
    r"^(?:结论|概览|要点|背景|补充说明|来源|参考资料|(?:各阶段)?(?:举办地点|赛程|场馆)(?:一览|概览)?|"
    r"分阶段安排|规格一览|参数对比|时间安排|关于.{1,12}的补充信息|补充信息)$"
)
WEB_TABLE_INTRO = re.compile(r"^(?:各阶段(?:举办)?地点如下|各阶段(?:场馆|安排)如下|具体安排如下)[：:]?$")
WEB_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?$")
WEB_TABLE_HEADERS = {
    "项目", "指标", "数值", "数据", "说明", "名称", "时间", "日期", "地点", "城市", "场馆", "来源", "依据",
    "阶段", "赛事阶段", "举办地点", "举办城市", "具体场馆", "规格", "参数", "结果", "备注",
}
KNOWLEDGE_BLOCK = re.compile(
    r"(?m)^引用：(knowledge://\d+#\d+)\n来源：[^\n]*\n内容：(.*?)\n【资料结束】", re.S
)
WEATHER_TOOLS = {"get_weather", "weather_current_weather"}
WEATHER_TERMS = ("天气", "气温", "下雨", "降雨")
WEATHER_TEMPERATURE = re.compile(r"气温\s*-?\d+(?:\.\d+)?\s*°C")
WEATHER_LOCATION = re.compile(r"^([^\n：]+)当前天气：")
WEATHER_FORECAST = re.compile(r"^([^\n：]+)天气预报（(\d{4}-\d{2}-\d{2})）：")
WEATHER_DAY_PART = re.compile(r"早上|上午|中午|下午|晚上|傍晚|凌晨|\d{1,2}\s*[点时]|\d{1,2}:\d{2}")
OTHER_WEATHER_TIME = re.compile(
    r"昨天|(?<!当)前天|昨晚|刚才|历史|过去|明天|明日|明早|明晚|明晨|后天|今晚|今早|今晨|"
    r"本周|这周|下周|上周|本月|这个月|下月|上月|月底|今年|去年|明年|近日|近几天|这几天|未来|接下来|"
    r"预报|预测|早上|上午|中午|下午|晚上|傍晚|凌晨|(?:星期|周|礼拜)[一二三四五六日天末\d]|"
    r"[零〇一二三四五六七八九十\d]+\s*(?:年|月|日|号|点|时|小时|天后|天前)|\d{1,4}[-/]\d{1,2}|\d{1,2}:\d{2}|"
    r"\b(?:tomorrow|yesterday|tonight|forecast|next|last|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.I,
)
CALCULATION_NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
CALCULATION_PREFIX = re.compile(r"^(?:(?:计算|两道算式的)?结果(?:分别)?(?:是|为|[:：]))\s*")


def _statements(text: str) -> set[str]:
    sentences = re.split(r"[。！？!?\n]+|\.(?=\s|$)", text)
    return {statement for sentence in sentences if (statement := " ".join(sentence.split()))}


def _is_web_tool(name: str | None) -> bool:
    return bool(name and (name == "web_search" or (
        name.startswith("search_") and name != "search_knowledge_base"
    )))


def _web_records(text: str):
    """支持内置工具的数组及 MCP 搜索的逐条 JSON 文本。"""
    decoder = json.JSONDecoder()
    while text := text.lstrip():
        try:
            value, end = decoder.raw_decode(text)
        except ValueError:
            return
        records = value if isinstance(value, list) else [value]
        for record in records:
            if not isinstance(record, dict):
                continue
            url, body = record.get("href") or record.get("url"), record.get("body")
            if not isinstance(url, str) or not URL_PATTERN.fullmatch(url):
                continue
            title, published = record.get("title"), record.get("date")
            yield (
                url,
                body if isinstance(body, str) else "",
                title if isinstance(title, str) else "",
                published if isinstance(published, str) else "",
            )
        text = text[end:]


def web_result_urls(messages: list[object]) -> set[str] | None:
    """统计本批成功搜索返回的不同链接，失败时不把错误内容计入。"""
    searches = [message for message in messages if isinstance(message, ToolMessage) and _is_web_tool(message.name)]
    if not searches:
        return None
    results = [message.content for message in searches if message.status == "success" and isinstance(message.content, str)]
    return {url for content in results for url, *_ in _web_records(content)} if results else None


def web_result_count(messages: list[object]) -> int | None:
    urls = web_result_urls(messages)
    return len(urls) if urls is not None else None


def web_result_records(content: str):
    for url, body, title, published in _web_records(content):
        yield {"href": url, "body": body, "title": title, "date": published}


def _web_summary_matches(claims: set[str], refs: set[str], evidence: dict[str, set[str]],
                         summary_sources: dict[str, set[str]], publication_dates: dict[str, set[str]],
                         *, strict_text_claims: bool = False) -> bool:
    def supports(ref: str, claim: str) -> bool:
        if strict_text_claims and ref not in summary_sources:
            return False
        if claim in evidence[ref]:
            return True
        if ref not in summary_sources:
            return False
        if strict_text_claims and not re.search(r"\d", claim):
            def normalize(text: str) -> str:
                return re.sub(r"[\s，,。.!！?？:：;；、\"'“”‘’（）()【】\[\]《》<>*`]+", "", text).casefold()

            normalized_claim = normalize(claim)
            return bool(normalized_claim) and any(
                normalized_claim in normalize(statement) for statement in summary_sources[ref]
            )
        return supports_numeric_claim(claim, summary_sources[ref], publication_dates[ref])

    return all(any(supports(ref, claim) for ref in refs) for claim in claims) and all(
        any(supports(ref, claim) for claim in claims) for ref in refs
    )


def _web_summary_blocks(answer: str):
    """为段落或列表项绑定内部链接及紧随其后的来源行。"""
    claims: set[str] = set()
    refs: set[str] = set()
    original: list[str] = []
    table_header = ""
    block_header = ""
    paragraph_ended = False
    lines = answer.splitlines()
    for index, line in enumerate(lines):
        line = line.strip()
        if not line:
            paragraph_ended = True
            table_header = ""
            continue
        item = WEB_LIST_ITEM.match(line)
        table_row = line.startswith("|") and line.endswith("|")
        line_refs = set(URL_PATTERN.findall(line))
        label = MARKDOWN_WEB_LINK.sub("", line)
        label = URL_PATTERN.sub("", label)
        if item:
            label = WEB_LIST_ITEM.sub("", label, count=1)
        label = WEB_SOURCE_ID.sub("", label)
        label = label.strip(" #*_`\t[]「」\"。；;，,、|-—")
        if WEB_TABLE_SEPARATOR.fullmatch(line):
            continue
        if not line_refs and (
            (line.startswith("#") and WEB_LAYOUT_HEADING.fullmatch(label.lstrip("🗺🏆📍📌️ ")))
            or WEB_TABLE_INTRO.fullmatch(label)
            or (table_row and index + 1 < len(lines) and WEB_TABLE_SEPARATOR.fullmatch(lines[index + 1].strip())
                and all(cell.strip(" *") in WEB_TABLE_HEADERS for cell in label.split("|")))
        ):
            if table_row:
                table_header = line + "\n" + lines[index + 1].strip()
            continue
        if line_refs and (not label or WEB_SOURCE_LABEL.fullmatch(label)):
            refs.update(line_refs)
            original.append(line)
            continue
        if not line_refs and (WEB_SOURCE_LABEL.fullmatch(label) or WEB_SECTION_LABEL.fullmatch(label)):
            continue
        if claims and (item or table_row or paragraph_ended or refs or line.startswith("#")):
            yield claims, refs, "\n".join(original), block_header
            claims, refs = set(), set()
            original = []
        if not original:
            block_header = table_header if table_row else ""
        if not table_row:
            table_header = ""
        claims.update(_statements(re.sub(r"\s*\|\s*", "，", label) if table_row else label))
        refs.update(line_refs)
        original.append(line)
        paragraph_ended = False
    if claims or refs:
        yield claims, refs, "\n".join(original), block_header


def supported_web_excerpt(question: str, answer: str, messages: list[object], *, strict_text_claims: bool) -> str:
    """保留通过同一证据校验的段落，供联网回答在整稿失败时使用。"""
    if not web_answer_sources(question, messages) or contains_secret(answer) or KNOWLEDGE_PATTERN.search(answer):
        return ""
    kept = []
    active_table = ""
    table_rows = []

    def flush_table():
        if table_rows:
            kept.append(active_table + "\n" + "\n".join(table_rows))
            table_rows.clear()

    for claims, refs, block, header in _web_summary_blocks(answer):
        if header != active_table:
            flush_table()
            active_table = header
        if not claims or not refs:
            continue
        candidate = header + "\n" + block if header else block
        if validate_answer(question, candidate, messages, requires_web_evidence=True, strict_text_claims=strict_text_claims)["ok"]:
            if header:
                row, *source_lines = block.splitlines()
                if source_lines:
                    links = " ".join(f"[来源]({url})" for url in sorted(refs - set(URL_PATTERN.findall(row))))
                    if links:
                        row = row[:-1].rstrip() + " " + links + " |"
                table_rows.append(row)
            elif not block.startswith("|"):
                kept.append(block)
    flush_table()
    if not kept:
        return ""
    excerpt = "\n\n".join(kept)
    return excerpt if validate_answer(
        question, excerpt, messages, requires_web_evidence=True, strict_text_claims=strict_text_claims,
    )["ok"] else ""


def _unsupported(reason: str) -> dict:
    return {"ok": False, "reason": reason, "replacement": "本轮资料不足以核实相关结论，暂时无法确认。你可以稍后重试。"}


def _calculation_evidence(messages: list[object]) -> tuple[list[str], bool]:
    expressions = {}
    results = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                expression = call["args"].get("expression")
                if call["name"] == "calculator" and isinstance(expression, str):
                    expressions[call["id"]] = expression.strip()
        elif isinstance(message, ToolMessage) and message.name == "calculator":
            expression = expressions.get(message.tool_call_id)
            key = ("expression", expression) if expression is not None else ("call", message.tool_call_id)
            value = message.content.strip() if isinstance(message.content, str) else ""
            results[key] = value if message.status == "success" and CALCULATION_NUMBER.fullmatch(value) else None
    return [value for value in results.values() if value is not None], None in results.values()


def _location_matches_question(question: str, place: str) -> bool:
    place = place.strip()
    if place.endswith("市") and len(place) > 2:
        place = place[:-1]
    # 英文地点按词匹配，中文允许地点紧邻“今天”“天气”等词。
    return bool(place and re.search(rf"(?<![A-Za-z]){re.escape(place)}(?![A-Za-z])", question, re.I))


def _forecast_date_matches_question(question: str, report_date: str) -> bool:
    if WEATHER_DAY_PART.search(question):
        return False
    target = Date.fromisoformat(report_date)
    today = Date.today()
    relative = {
        "今天": today, "今日": today,
        "明天": today + timedelta(days=1), "明日": today + timedelta(days=1),
        "后天": today + timedelta(days=2),
    }
    matched_relative = [expected for term, expected in relative.items() if term in question]
    if matched_relative:
        return target in matched_relative
    iso_dates = {Date.fromisoformat(value) for value in re.findall(r"\d{4}-\d{2}-\d{2}", question)}
    if iso_dates:
        return target in iso_dates
    full_dates = {
        Date(int(year), int(month), int(day))
        for year, month, day in re.findall(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]", question)
    }
    if full_dates:
        return target in full_dates
    short_dates = {(int(month), int(day)) for month, day in re.findall(r"(?<!年)(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]", question)}
    if short_dates:
        return (target.month, target.day) in short_dates
    weekday = re.search(r"(?:星期|周|礼拜)([一二三四五六日天末1-7])", question)
    if weekday:
        values = {"一": 0, "1": 0, "二": 1, "2": 1, "三": 2, "3": 2, "四": 3, "4": 3,
                  "五": 4, "5": 4, "六": 5, "6": 5, "日": 6, "天": 6, "末": 6, "7": 6}
        return target.weekday() == values[weekday[1]]
    # “未来几天”“本周天气”等范围问题允许使用工具返回的多个带日期预报行。
    return bool(re.search(r"预报|预测|未来|接下来|近几天|这几天|本周|这周|下周", question)) or not OTHER_WEATHER_TIME.search(question)


def _weather_reports_for_question(question: str, content: str) -> set[str]:
    matched = set()
    for statement in _statements(content):
        if not WEATHER_TEMPERATURE.search(statement):
            continue
        current = WEATHER_LOCATION.match(statement)
        if current and not OTHER_WEATHER_TIME.search(question) and _location_matches_question(question, current[1]):
            matched.add(statement)
            continue
        forecast = WEATHER_FORECAST.match(statement)
        if (
            forecast
            and _location_matches_question(question, forecast[1])
            and _forecast_date_matches_question(question, forecast[2])
        ):
            matched.add(statement)
    return matched


def weather_tool_answer(question: str, messages: list[object]) -> str:
    """从本轮内置天气工具的有效结果中提取完整报告，保留行顺序与标点。"""
    if not any(term in question for term in WEATHER_TERMS):
        return ""
    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    if any(message.name != "get_weather" for message in tool_messages):
        return ""

    reports = {}
    for message in tool_messages:
        if message.status != "success" or not isinstance(message.content, str):
            continue
        for line in message.content.splitlines():
            statements = _statements(line)
            if statements and statements == _weather_reports_for_question(question, line):
                reports.setdefault(line.strip(), None)
    return "\n".join(reports)


def weather_activity_advice(report: str) -> str:
    """从已核验的预报生成活动建议，保留预报的时间粒度。"""
    if re.search(r"雷暴|雷雨|冰雹|暴雨|大雨|强阵雨|大雪", report):
        return "不建议按原计划进行长时间户外活动。预报显示有较明显的天气风险，建议改为室内活动，出发前查看当地预警。"
    probabilities = [int(value) for value in re.findall(r"最大降水概率\s*(\d+)%", report)]
    if re.search(r"雨|雪", report) or max(probabilities, default=0) >= 60:
        return "可以安排出游，但不太适合长时间在户外。建议优先安排室内活动，带好雨具，户外行程可结合临近时段的预报调整。"
    return "从现有天气信息看，可以考虑安排户外活动。出发前再确认活动时段的预报；徒步、露营等还需要查看当地预警和现场条件。"


def web_answer_sources(question: str, messages: list[object]) -> list[dict[str, str]]:
    """选取本轮可供普通联网回答使用的来源。"""
    if requested_knowledge_query(question) is not None or any(
        term in question for term in (*STRICT_WEB_TERMS, *HIGH_RISK_TERMS, *WEATHER_TERMS)
    ):
        return []
    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    if any(not _is_web_tool(message.name) for message in tool_messages):
        return []
    queries = {
        call["id"]: str(call["args"].get("query") or "")
        for message in messages if isinstance(message, AIMessage)
        for call in message.tool_calls if _is_web_tool(call["name"])
    }
    entries: dict[str, dict[str, str]] = {}
    for message in reversed(tool_messages):
        if message.status != "success" or not isinstance(message.content, str):
            continue
        for url, body, title, published in _web_records(message.content):
            title = " ".join(title.split())[:200] if not contains_secret(title) else ""
            published = published[:80] if not contains_secret(published) else ""
            # 正文可能引用其它页面，引用仍绑定到当前结果的链接。
            body = REFERENCE_PATTERN.sub("", body)
            if contains_secret(body) or contains_secret(url):
                continue
            excerpt = extract_web_passages(body, question, query=queries.get(message.tool_call_id, ""))
            if not excerpt and not title:
                continue
            try:
                if not urlparse(url).hostname:
                    continue
            except ValueError:
                continue
            entry = {
                "url": url, "excerpt": excerpt or title, "date": published, "title": title,
                "kind": "摘录" if excerpt else "仅标题",
            }
            previous = entries.get(url)
            if previous is None or (excerpt and previous["kind"] == "仅标题"):
                entries[url] = entry
            elif excerpt and excerpt != previous["excerpt"]:
                # 多次搜索可能命中同一网页的不同段落。保留一个编号，合并再选。
                previous["excerpt"] = extract_web_passages(
                    previous["excerpt"] + "\n" + excerpt, question, query=queries.get(message.tool_call_id, ""),
                )
    return list(entries.values())


def render_web_citations(answer: str, sources: list[dict[str, str]]) -> str:
    """链接只能由本轮来源编号渲染。"""
    if URL_PATTERN.search(answer) or KNOWLEDGE_PATTERN.search(answer) or not WEB_SOURCE_ID.search(answer):
        return ""
    links = {str(index): entry["url"] for index, entry in enumerate(sources, 1)}
    if any(source_id not in links for source_id in WEB_SOURCE_ID.findall(answer)):
        return ""
    return WEB_SOURCE_ID.sub(lambda match: f"[来源 {match[1]}]({links[match[1]]})", answer)


def renumber_web_citations(answer: str) -> str:
    """在校验完成后按引用的首次出现顺序重新编号。"""
    numbers: dict[str, int] = {}

    def replace(match: re.Match[str]) -> str:
        url = match[1]
        return f"[来源 {numbers.setdefault(url, len(numbers) + 1)}]({url})"

    return NUMBERED_WEB_LINK.sub(replace, answer)


def format_web_answer(answer: str, sources: list[dict[str, str]]) -> str:
    """保留正文，只把本轮来源变成可点击的引用。"""
    links = {str(index): source["url"] for index, source in enumerate(sources, 1)}
    allowed = set(links.values())
    answer = MARKDOWN_WEB_LINK.sub(lambda match: match[0] if match[2] in allowed else match[1], answer)
    answer = URL_PATTERN.sub(lambda match: match[0] if match[0] in allowed else "", answer)
    return WEB_SOURCE_ID.sub(
        lambda match: f"[来源 {match[1]}]({links[match[1]]})" if match[1] in links else "", answer,
    )


def cited_web_sources(answer: str, messages: list[object]) -> list[dict[str, str]]:
    """从已交付正文中提取可回溯到本轮成功工具的来源。"""
    links = dict.fromkeys(URL_PATTERN.findall(answer))
    if not links:
        return []
    records = {}
    for message in messages:
        if not isinstance(message, ToolMessage) or message.status != "success" or not _is_web_tool(message.name):
            continue
        if not isinstance(message.content, str):
            continue
        for url, _, title, published in _web_records(message.content):
            if url not in links or url in records:
                continue
            try:
                domain = urlparse(url).hostname
            except ValueError:
                continue
            if domain:
                title = "" if contains_secret(title) else " ".join(title.split())[:200]
                records[url] = {"url": url, "title": title or domain, "domain": domain}
    return [records[url] for url in links if url in records]


def news_tool_answer(question: str, messages: list[object], *, requires_web_evidence: bool = False) -> str:
    """修订仍无法核验时，展示本轮实际取得的检索线索。"""
    if not requires_web_evidence and "新闻" not in question:
        return ""
    entries = web_answer_sources(question, messages)
    if not entries:
        return ""
    lines = [f"{index}. {entry['excerpt']}\n来源：{entry['url']}" for index, entry in enumerate(entries, 1)]
    return "相关报道\n\n" + "\n\n".join(lines)


def validate_output_text(answer: str) -> dict:
    """检查回答能否安全作为正文交付。"""
    if not answer.strip() or len(answer) > 12000:
        return {"ok": False, "reason": "回答为空或超出安全长度", "replacement": "当前无法生成可安全验证的回答，请缩小问题范围后重试。"}
    if contains_secret(answer):
        return {"ok": False, "reason": "回答包含疑似敏感凭据", "replacement": "为保护安全，我不能在回答中展示访问密钥、令牌或密码等敏感凭据。"}
    return {"ok": True, "reason": "通过基础输出检查", "replacement": ""}


def validate_answer(question: str, answer: str, messages: list[object], *,
                    requires_web_evidence: bool = False, strict_text_claims: bool = False) -> dict:
    """校验引用归属、原文支持及高风险领域。"""
    result = validate_output_text(answer)
    if not result["ok"]:
        return result
    lowered = question.lower()
    if any(term in lowered for term in HIGH_RISK_TERMS):
        return {"ok": False, "reason": "高风险领域需要人工或权威专业来源复核", "replacement": "该问题涉及医疗、法律或金融等高风险领域。我不能提供未经专业核验的结论；请咨询持证专业人士或提供权威来源后再协助解读。"}
    calculation_values, unresolved_calculation = _calculation_evidence(messages)
    if unresolved_calculation:
        return {
            "ok": False,
            "reason": "本轮有计算未取得有效结果",
            "replacement": "本轮有表达式未能完成计算，暂时无法确认结果。请检查表达式后重试。",
        }

    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    knowledge_only = requested_knowledge_query(question) is not None
    evidence: dict[str, set[str]] = {}
    summary_sources: dict[str, set[str]] = {}
    publication_dates: dict[str, set[str]] = {}
    weather_reports: set[str] = set()
    unmatched_weather = False
    for message in tool_messages:
        if knowledge_only and message.name != "search_knowledge_base":
            continue
        if message.status != "success" or not isinstance(message.content, str):
            continue
        content = message.content
        if message.name in WEATHER_TOOLS and WEATHER_TEMPERATURE.search(content):
            matched_reports = _weather_reports_for_question(question, content)
            if matched_reports:
                weather_reports.update(matched_reports)
            else:
                unmatched_weather = True
        elif _is_web_tool(message.name):
            for url, body, title, published in _web_records(content):
                evidence.setdefault(url, set()).update(_statements(body))
                if body.strip():
                    summary_sources.setdefault(url, set()).update(_statements(body))
                    publication_dates.setdefault(url, set()).add(published)
                else:
                    evidence[url].update(_statements(title))
        elif message.name == "search_knowledge_base":
            for citation, body in KNOWLEDGE_BLOCK.findall(content):
                evidence.setdefault(citation, set()).update(_statements(body))

    evidence_urls = {ref for ref, statements in evidence.items() if statements and ref.startswith("http")}
    evidence_citations = {ref for ref, statements in evidence.items() if statements and ref.startswith("knowledge://")}
    used_knowledge_tool = any(message.name == "search_knowledge_base" for message in tool_messages)
    used_web_tool = any(_is_web_tool(message.name) for message in tool_messages)
    citations = set(KNOWLEDGE_PATTERN.findall(answer))
    if (knowledge_only and not citations) or ((used_knowledge_tool or citations) and not evidence_citations) or citations - evidence_citations:
        return _unsupported("没有可对应的知识库证据")

    needs_freshness = requires_web_evidence or any(term in question for term in FRESHNESS_TERMS)
    weather_requested = any(term in question for term in WEATHER_TERMS)
    strict_web = any(term in question for term in STRICT_WEB_TERMS)
    allow_web_summary = not (strict_web or weather_requested or knowledge_only or used_knowledge_tool or calculation_values)
    if not knowledge_only and (weather_requested or unmatched_weather) and not weather_reports:
        if OTHER_WEATHER_TIME.search(question):
            return {"ok": False, "reason": "天气工具未返回匹配地点与时段的预报", "replacement": "天气工具没有取得与地点和日期匹配的有效预报，暂时无法确认。"}
        return _unsupported("未取得与问题地点对应的当前天气数据")
    urls = set(URL_PATTERN.findall(answer))
    if requires_web_evidence and not (urls & evidence_urls):
        return {
            "ok": False, "reason": "具体事件缺少本轮可引用的联网证据",
            "replacement": "目前没有足够的可引用资料核实这件事，暂时无法确认是否发生，也无法判断原因。",
        }
    if not knowledge_only and (needs_freshness or used_web_tool) and not (evidence_urls or weather_reports):
        return _unsupported("没有可用的联网或天气工具证据")
    failed_urls = {
        url for message in tool_messages if message.status != "success"
        for url in URL_PATTERN.findall(str(message.content))
    }
    if (urls & failed_urls) - evidence_urls:
        return _unsupported("回答引用了失败工具中的链接")

    if urls and evidence_urls:
        if urls - evidence_urls:
            return _unsupported("回答引用了工具结果中不存在的链接")
        try:
            domains = [urlparse(url).hostname for url in urls]
        except ValueError:
            return _unsupported("来源链接格式无效")
        if not all(domains):
            return _unsupported("来源链接缺少域名")
        if (strict_web or ("新闻" in question and not allow_web_summary)) and len(set(domains)) < 2:
            return _unsupported("关键时效事实缺少多源交叉验证")

    calculation_line = "计算结果：" + "、".join(calculation_values) if calculation_values else ""
    if calculation_line and not (evidence_urls or evidence_citations or weather_reports):
        claimed_values = CALCULATION_PREFIX.sub("", answer.strip().rstrip("。！？!"), count=1)
        if re.split(r"\s*(?:[、，,]|和)\s*", claimed_values) != calculation_values:
            replacement = calculation_line if len(calculation_line) <= 12000 else "计算结果过长，无法在本轮完整展示。"
            return {"ok": False, "reason": "回答与计算工具结果不一致或无法逐项核验", "replacement": replacement}

    if allow_web_summary and evidence_urls and not evidence_citations and not weather_reports and not citations:
        has_claim = False
        for claims, refs, _, _ in _web_summary_blocks(answer):
            if not claims or not refs:
                return _unsupported("联网摘要段落缺少对应来源或正文")
            if not _web_summary_matches(
                claims, refs, evidence, summary_sources, publication_dates,
                strict_text_claims=strict_text_claims,
            ):
                return _unsupported("联网摘要的数字、单位或限定条件未获对应正文支持，或超出标题证据")
            has_claim = True
        if not has_claim:
            return _unsupported("回答缺少可核对的结论")
    elif evidence_urls or evidence_citations or weather_reports:
        valid_refs = evidence_urls | evidence_citations
        found_claim = False
        found_weather = False
        for line in answer.splitlines():
            refs = set(REFERENCE_PATTERN.findall(line))
            claim_text = MARKDOWN_WEB_LINK.sub("", line)
            claim_text = REFERENCE_PATTERN.sub("", claim_text).strip()
            claim_text = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", claim_text)
            claims = _statements(claim_text.strip(' \t[]()「」“”"'))
            if refs - valid_refs or (refs and not claims):
                return _unsupported("引用与本轮资料无法对应")
            knowledge_refs = refs & evidence_citations
            if knowledge_refs and any(not claims <= evidence[ref] for ref in knowledge_refs):
                return _unsupported("知识库引用须逐条支持同一行的原文陈述")
            supported = set().union(*(evidence[ref] for ref in refs)) if refs else weather_reports | {calculation_line}
            if claims - supported or any(not (claims & evidence[ref]) for ref in refs):
                return _unsupported("引用资料不支持回答中的原句")
            found_claim |= bool(claims)
            found_weather |= not refs and bool(claims & weather_reports)
        if not found_claim or (not knowledge_only and weather_requested and not found_weather):
            return _unsupported("回答缺少可核对的结论")

    return {"ok": True, "reason": "通过基础证据与风险校验", "replacement": ""}
