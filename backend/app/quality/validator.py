"""无需模型的强制回答校验；校验失败时宁可降级也不编造。"""

import re
from urllib.parse import urlparse

FRESHNESS_TERMS = ("最新", "今天", "当前", "实时", "价格", "新闻", "天气", "政策")
HIGH_RISK_TERMS = ("诊断", "用药", "疾病", "法律", "起诉", "合同", "投资", "股票", "理财", "贷款")
URL_PATTERN = re.compile(r"https?://[^\s)\]}>]+")
KNOWLEDGE_PATTERN = re.compile(r"knowledge://\d+#\d+")
FACTUAL_PATTERN = re.compile(r"\d|根据|显示|预计|将|为|是|有")


def _factual_sentences(answer: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"[。！？!？\n]+", answer) if len(sentence.strip()) >= 8 and FACTUAL_PATTERN.search(sentence)]


def validate_answer(question: str, answer: str, messages: list[object]) -> dict:
    """检查高风险、时效性和网络事实的最低证据要求。"""
    lowered = question.lower()
    if not answer.strip() or len(answer) > 12000:
        return {"ok": False, "reason": "回答为空或超出安全长度", "replacement": "当前无法生成可安全验证的回答，请缩小问题范围后重试。"}
    if any(term in lowered for term in HIGH_RISK_TERMS):
        return {"ok": False, "reason": "高风险领域需要人工或权威专业来源复核", "replacement": "该问题涉及医疗、法律或金融等高风险领域。我不能提供未经专业核验的结论；请咨询持证专业人士或提供权威来源后再协助解读。"}

    tool_names = {getattr(message, "name", "") for message in messages}
    has_knowledge_evidence = "search_knowledge_base" in tool_names
    has_web_evidence = "web_search" in tool_names or any(
        name.startswith("search_") and name != "search_knowledge_base" for name in tool_names
    )
    has_weather_evidence = "get_weather" in tool_names or "weather_current_weather" in tool_names
    needs_freshness = any(term in question for term in FRESHNESS_TERMS)
    if needs_freshness and not (has_web_evidence or has_weather_evidence):
        return {"ok": False, "reason": "时效性信息缺少工具证据", "replacement": "这个问题包含时效性信息。为避免提供过期内容，请允许我先联网检索或查询权威工具结果。"}

    if has_web_evidence:
        urls = URL_PATTERN.findall(answer)
        evidence_urls = {
            url
            for message in messages
            if getattr(message, "name", "") == "web_search" or getattr(message, "name", "").startswith("search_")
            for url in URL_PATTERN.findall(str(getattr(message, "content", "")))
        }
        domains = {urlparse(url).netloc.lower() for url in urls if urlparse(url).netloc}
        if not urls:
            return {"ok": False, "reason": "联网结论未附来源链接", "replacement": "我已检索到相关资料，但当前回答未能提供可追溯链接；请重新查询后再输出结论。"}
        if not set(urls).issubset(evidence_urls):
            return {"ok": False, "reason": "回答引用了工具结果中不存在的链接", "replacement": "回答中的来源无法与工具结果对应；我不会输出不可追溯的引用。"}
        if any(term in question for term in ("政策", "法律", "价格", "新闻")) and len(domains) < 2:
            return {"ok": False, "reason": "关键时效事实缺少多源交叉验证", "replacement": "该结论需要至少两个独立来源交叉核对；当前证据不足，我不会给出确定性结论。"}
        if any(not any(url in sentence for url in urls) for sentence in _factual_sentences(answer)):
            return {"ok": False, "reason": "存在未附来源的事实性陈述", "replacement": "回答包含无法逐条追溯的事实性陈述；请重新检索并在每个关键结论旁标注来源。"}
    if has_knowledge_evidence:
        citations = KNOWLEDGE_PATTERN.findall(answer)
        evidence_citations = {
            citation
            for message in messages
            if getattr(message, "name", "") == "search_knowledge_base"
            for citation in KNOWLEDGE_PATTERN.findall(str(getattr(message, "content", "")))
        }
        if not citations:
            return {"ok": False, "reason": "知识库结论未附可追溯引用", "replacement": "我检索到了内部资料，但回答未附知识库引用，因此不会输出无法追溯的结论。"}
        if not set(citations).issubset(evidence_citations):
            return {"ok": False, "reason": "知识库引用无法与检索片段对应", "replacement": "回答中的知识库引用无法与实际检索片段对应；我不会输出不可验证的结论。"}
        if any(not KNOWLEDGE_PATTERN.search(sentence) for sentence in _factual_sentences(answer)):
            return {"ok": False, "reason": "存在未附知识库引用的事实性陈述", "replacement": "回答包含无法逐条追溯的内部事实；请在每个关键结论旁附知识库引用。"}
    return {"ok": True, "reason": "通过基础证据与风险校验", "replacement": ""}
