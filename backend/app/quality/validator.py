"""校验回答的来源、数字及需要逐句核对的证据。"""

import json
import re
from datetime import date as Date, timedelta
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, ToolMessage

from backend.app.content_safety import contains_secret
from backend.app.intent.policy import requested_knowledge_query

FRESHNESS_TERMS = ("最新", "今天", "当前", "实时", "价格", "新闻", "天气", "政策")
INTERNAL_FRESHNESS_TERMS = ("最新", "今天", "当前", "实时", "现行", "最近")
STRICT_WEB_TERMS = ("政策", "法律", "价格")
HIGH_RISK_TERMS = ("诊断", "用药", "疾病", "法律", "起诉", "合同", "投资", "股票", "理财", "贷款")
URL_PATTERN = re.compile(r"""https?://[^\s<>()\[\]{}"'，。！？；、]+""")
KNOWLEDGE_PATTERN = re.compile(r"knowledge://\d+#\d+(?!\d)")
REFERENCE_PATTERN = re.compile(f"{URL_PATTERN.pattern}|{KNOWLEDGE_PATTERN.pattern}")
MARKDOWN_WEB_LINK = re.compile(rf"(?<!!)\[([^\]\n]*)\]\(({URL_PATTERN.pattern})\)")
WEB_NUMBER = re.compile(r"\d+(?:[.,]\d+)*%?")
WEB_SOURCE_ID = re.compile(r"\[S([1-9]\d*)\]")
WEB_LIST_ITEM = re.compile(r"^(?:[-*+•]|\d+[.)、]|[一二三四五六七八九十]+、)\s*")
WEB_SOURCE_LABEL = re.compile(r"(?:参考(?:资料|文献)?|消息来源|新闻来源|参考来源|来源链接|来源|链接|出处|报道|sources?|references?)[：:]?", re.I)
WEB_SECTION_LABEL = re.compile(r"(?:以下是)?(?:近期|最近|最新)?(?:新闻(?:摘要|动态|速览|要点)?|动态|简报|摘要|主要动态|主要消息|消息汇总|相关报道)(?:如下)?[：:]?")
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


def _web_numbers(text: str) -> set[str]:
    return {number.replace(",", "") for number in WEB_NUMBER.findall(text.replace("％", "%"))}


def _web_summary_matches(claims: set[str], refs: set[str], evidence: dict[str, set[str]], summary_sources: dict[str, set[str]]) -> bool:
    def supports(ref: str, claim: str) -> bool:
        if claim in evidence[ref]:
            return True
        return ref in summary_sources and _web_numbers(claim) <= summary_sources[ref]

    return all(any(supports(ref, claim) for ref in refs) for claim in claims) and all(
        any(supports(ref, claim) for claim in claims) for ref in refs
    )


def _web_summary_blocks(answer: str):
    """为段落或列表项绑定内部链接及紧随其后的来源行。"""
    claims: set[str] = set()
    refs: set[str] = set()
    paragraph_ended = False
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            paragraph_ended = True
            continue
        item = WEB_LIST_ITEM.match(line)
        line_refs = set(URL_PATTERN.findall(line))
        label = MARKDOWN_WEB_LINK.sub("", line)
        label = URL_PATTERN.sub("", label)
        if item:
            label = WEB_LIST_ITEM.sub("", label, count=1)
        label = label.strip(" #*_`\t[]()（）「」\"。；;，,、|-—")
        if line_refs and (not label or WEB_SOURCE_LABEL.fullmatch(label)):
            refs.update(line_refs)
            continue
        if not line_refs and (WEB_SOURCE_LABEL.fullmatch(label) or WEB_SECTION_LABEL.fullmatch(label)):
            continue
        if claims and (item or paragraph_ended or refs or line.startswith("#")):
            yield claims, refs
            claims, refs = set(), set()
        claims.update(_statements(label))
        refs.update(line_refs)
        paragraph_ended = False
    if claims or refs:
        yield claims, refs


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
    entries: dict[str, dict[str, str]] = {}
    for message in tool_messages:
        if message.status != "success" or not isinstance(message.content, str):
            continue
        for url, body, title, published in _web_records(message.content):
            excerpt = " ".join((body.strip() or title).split())
            if not excerpt or len(excerpt) > 600 or contains_secret(excerpt) or REFERENCE_PATTERN.search(excerpt):
                continue
            try:
                if not urlparse(url).hostname:
                    continue
            except ValueError:
                continue
            entries.setdefault(url, {
                "url": url, "excerpt": excerpt, "date": published[:80], "title": title[:200],
                "kind": "摘要" if body.strip() else "仅标题",
            })
    return list(entries.values())[:8]


def render_web_citations(answer: str, sources: list[dict[str, str]]) -> str:
    """链接只能由本轮来源编号渲染。"""
    if URL_PATTERN.search(answer) or KNOWLEDGE_PATTERN.search(answer) or not WEB_SOURCE_ID.search(answer):
        return ""
    links = {str(index): entry["url"] for index, entry in enumerate(sources, 1)}
    if any(source_id not in links for source_id in WEB_SOURCE_ID.findall(answer)):
        return ""
    return WEB_SOURCE_ID.sub(lambda match: f"[来源 {match[1]}]({links[match[1]]})", answer)


def news_tool_answer(question: str, messages: list[object], *, requires_web_evidence: bool = False) -> str:
    """修订仍无法核验时，展示本轮实际取得的检索线索。"""
    if not requires_web_evidence and "新闻" not in question:
        return ""
    entries = web_answer_sources(question, messages)
    if not entries:
        return ""
    lines = [f"{index}. {entry['excerpt']}\n来源：{entry['url']}" for index, entry in enumerate(entries[:3], 1)]
    return "相关报道\n\n" + "\n\n".join(lines)


def validate_answer(question: str, answer: str, messages: list[object], *, requires_web_evidence: bool = False) -> dict:
    """校验引用归属、原文支持及高风险领域。"""
    lowered = question.lower()
    if not answer.strip() or len(answer) > 12000:
        return {"ok": False, "reason": "回答为空或超出安全长度", "replacement": "当前无法生成可安全验证的回答，请缩小问题范围后重试。"}
    if contains_secret(answer):
        return {"ok": False, "reason": "回答包含疑似敏感凭据", "replacement": "为保护安全，我不能在回答中展示访问密钥、令牌或密码等敏感凭据。"}
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
                    summary_sources.setdefault(url, set()).update(_web_numbers("\n".join([body, title, published])))
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

    needs_freshness = any(term in question for term in FRESHNESS_TERMS)
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
        for claims, refs in _web_summary_blocks(answer):
            if not claims or not refs:
                return _unsupported("联网摘要段落缺少对应来源或正文")
            if not _web_summary_matches(claims, refs, evidence, summary_sources):
                return _unsupported("联网摘要包含来源无法核对的数字，或超出标题证据")
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
            supported = set().union(*(evidence[ref] for ref in refs)) if refs else weather_reports | {calculation_line}
            if claims - supported or any(not (claims & evidence[ref]) for ref in refs):
                return _unsupported("引用资料不支持回答中的原句")
            found_claim |= bool(claims)
            found_weather |= not refs and bool(claims & weather_reports)
        if not found_claim or (not knowledge_only and weather_requested and not found_weather):
            return _unsupported("回答缺少可核对的结论")

    return {"ok": True, "reason": "通过基础证据与风险校验", "replacement": ""}
