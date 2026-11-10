"""按来源原句执行强制回答校验。"""

import json
import re
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, ToolMessage

from backend.app.content_safety import contains_secret
from backend.app.intent.policy import requested_knowledge_query

FRESHNESS_TERMS = ("最新", "今天", "当前", "实时", "价格", "新闻", "天气", "政策")
INTERNAL_FRESHNESS_TERMS = ("最新", "今天", "当前", "实时", "现行", "最近")
HIGH_RISK_TERMS = ("诊断", "用药", "疾病", "法律", "起诉", "合同", "投资", "股票", "理财", "贷款")
URL_PATTERN = re.compile(r"""https?://[^\s<>()\[\]{}"'，。！？；、]+""")
KNOWLEDGE_PATTERN = re.compile(r"knowledge://\d+#\d+(?!\d)")
REFERENCE_PATTERN = re.compile(f"{URL_PATTERN.pattern}|{KNOWLEDGE_PATTERN.pattern}")
KNOWLEDGE_BLOCK = re.compile(
    r"(?m)^引用：(knowledge://\d+#\d+)\n来源：[^\n]*\n内容：(.*?)\n【资料结束】", re.S
)
WEATHER_TOOLS = {"get_weather", "weather_current_weather"}
WEATHER_TEMPERATURE = re.compile(r"气温\s*-?\d+(?:\.\d+)?\s*°C")
WEATHER_LOCATION = re.compile(r"^([^\n：]+)当前天气：")
OTHER_WEATHER_TIME = re.compile(
    r"昨天|前天|昨晚|刚才|历史|过去|明天|明日|明早|明晚|明晨|后天|今晚|今早|今晨|"
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
            url, body = record.get("href"), record.get("body")
            if isinstance(url, str) and isinstance(body, str) and URL_PATTERN.fullmatch(url):
                yield url, body
        text = text[end:]


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


def _weather_matches_question(question: str, content: str) -> bool:
    location = WEATHER_LOCATION.match(content)
    if not location or OTHER_WEATHER_TIME.search(question):
        return False
    place = location[1].strip()
    if place.endswith("市") and len(place) > 2:
        place = place[:-1]
    # 英文地点按词匹配，中文允许地点紧邻“今天”“天气”等词。
    return bool(place and re.search(rf"(?<![A-Za-z]){re.escape(place)}(?![A-Za-z])", question, re.I))


def validate_answer(question: str, answer: str, messages: list[object]) -> dict:
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
    weather_reports: set[str] = set()
    unmatched_weather = False
    for message in tool_messages:
        if knowledge_only and message.name != "search_knowledge_base":
            continue
        if message.status != "success" or not isinstance(message.content, str):
            continue
        content = message.content
        if message.name in WEATHER_TOOLS and WEATHER_TEMPERATURE.search(content):
            if _weather_matches_question(question, content):
                weather_reports.update(_statements(content))
            else:
                unmatched_weather = True
        elif _is_web_tool(message.name):
            for url, body in _web_records(content):
                evidence.setdefault(url, set()).update(_statements(body))
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
    weather_requested = any(term in question for term in ("天气", "气温", "下雨", "降雨"))
    if not knowledge_only and (weather_requested or unmatched_weather) and not weather_reports:
        if OTHER_WEATHER_TIME.search(question):
            return {"ok": False, "reason": "当前天气工具无法核验请求时段", "replacement": "天气工具只能核验当前实况，暂时无法确认你询问时段的天气。"}
        return _unsupported("未取得与问题地点对应的当前天气数据")
    if not knowledge_only and (needs_freshness or used_web_tool) and not (evidence_urls or weather_reports):
        return _unsupported("没有可用的联网或天气工具证据")

    urls = set(URL_PATTERN.findall(answer))
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
        if any(term in question for term in ("政策", "法律", "价格", "新闻")) and len(set(domains)) < 2:
            return _unsupported("关键时效事实缺少多源交叉验证")

    calculation_line = "计算结果：" + "、".join(calculation_values) if calculation_values else ""
    if calculation_line and not (evidence_urls or evidence_citations or weather_reports):
        claimed_values = CALCULATION_PREFIX.sub("", answer.strip().rstrip("。！？!"), count=1)
        if re.split(r"\s*(?:[、，,]|和)\s*", claimed_values) != calculation_values:
            replacement = calculation_line if len(calculation_line) <= 12000 else "计算结果过长，无法在本轮完整展示。"
            return {"ok": False, "reason": "回答与计算工具结果不一致或无法逐项核验", "replacement": replacement}

    if evidence_urls or evidence_citations or weather_reports:
        valid_refs = evidence_urls | evidence_citations
        found_claim = False
        found_weather = False
        for line in answer.splitlines():
            refs = set(REFERENCE_PATTERN.findall(line))
            claim_text = REFERENCE_PATTERN.sub("", line).strip()
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
