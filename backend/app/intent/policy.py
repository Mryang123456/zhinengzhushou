"""可解释、保守的意图路由：支持同义表达和受控复合任务。"""

import re
from difflib import SequenceMatcher

from backend.app.skills.registry import Skill

ALIASES = {
    "联网调研": ("网上查", "帮我找", "搜一下", "资料收集", "市场研究", "新闻动态", "最新消息"),
    "出行规划": ("去哪玩", "旅行安排", "旅游攻略", "路线安排", "行程安排", "出游"),
}
MATH_PATTERN = re.compile(r"^[\d\s+\-*/%().]+$|\b(计算|算一下|求值)\b", re.I)
CASUAL_PATTERN = re.compile(r"^(你好|嗨|谢谢|帮助|你能做什么)[！!？?。\s]*$")
MEMORY_PATTERN = re.compile(r"记住|回忆|想起|删除.*记忆|忘记")
KNOWLEDGE_SOURCE = r"(?:知识库|内部(?:制度|政策|规定|资料|文档|手册)|公司(?:的)?(?:制度|手册))"
KNOWLEDGE_REQUEST = re.compile(
    rf"(?:从|在|到|根据|依据|基于|按照|按|参考|使用|查阅|查询|检索|搜索|查一下|查)\s*(?:(?:我们|公司|企业|内部)(?:的)?\s*)?"
    rf"{KNOWLEDGE_SOURCE}(?:\s*(?:里面|里|中|内))?|{KNOWLEDGE_SOURCE}\s*(?:里面|里|中|内)"
)
KNOWLEDGE_QUERY_PREFIX = re.compile(r"^(?:(?:请(?=\s|帮我|查|检索|搜索|看看)|帮我|查一下|查找|查查|查询|查|检索|搜索|看看|的)[\s，,:：]*)+")


def requested_knowledge_query(question: str) -> str | None:
    """明确指定内部来源时，剥离路由措辞，保留供检索的内容。"""
    match = KNOWLEDGE_REQUEST.search(question)
    if not match or re.search(r"(?:不要|不用|无需|禁止|别|不)\s*(?:再|去)?\s*$", question[:match.start()]):
        return None
    before = question[:match.start()].strip()
    if re.fullmatch(r"(?:请\s*)?(?:帮我\s*)?", before):
        before = ""
    after = question[match.end():].lstrip(" \t，,:：")
    query = " ".join(part for part in (before, KNOWLEDGE_QUERY_PREFIX.sub("", after)) if part)
    return query.strip(" \t，,。！!？?：:")


def _score(skill: Skill, normalized: str) -> float:
    terms = (*skill.triggers, *ALIASES.get(skill.name, ()))
    exact = sum(1 for term in terms if term in normalized)
    fuzzy = max((SequenceMatcher(None, term, normalized).ratio() for term in terms if len(term) >= 3), default=0.0)
    return exact * 2 + (0.75 if exact == 0 and fuzzy >= 0.62 else 0.0)


def classify(question: str, skills: list[Skill]) -> dict:
    knowledge_query = requested_knowledge_query(question)
    if knowledge_query is not None:
        if not knowledge_query:
            return {"status": "clarify", "reason": "缺少知识库查询内容", "skills": [], "confidence": 0.97, "question": "你想在知识库里查什么？"}
        return {
            "status": "allow", "reason": "用户明确指定内部知识来源", "skills": [], "confidence": 0.97,
            "question": "", "knowledge_query": knowledge_query,
        }
    normalized = re.sub(r"\s+", "", question.lower())
    if MATH_PATTERN.search(normalized) or CASUAL_PATTERN.match(normalized) or MEMORY_PATTERN.search(normalized):
        return {"status": "allow", "reason": "基础任务直通", "skills": [], "confidence": 0.95, "question": ""}
    matched = [(skill, score) for skill, score in ((skill, _score(skill, normalized)) for skill in skills) if score > 0]
    if not matched:
        return {"status": "allow", "reason": "通用问答", "skills": [], "confidence": 0.0, "question": "", "general": True}
    matched.sort(key=lambda item: item[1], reverse=True)
    best_skill, best_score = matched[0]
    secondary = [skill for skill, score in matched[1:] if score >= best_score * 0.8]
    compound = bool(re.search(r"并且|同时|再|以及|和", normalized))
    selected = [best_skill] + (secondary[:1] if compound else [])
    if secondary and not compound:
        return {"status": "clarify", "reason": "多个意图置信度接近", "skills": [], "confidence": 0.55, "question": f"你的问题可能同时涉及“{best_skill.name}”和“{secondary[0].name}”。请说明优先目标。"}
    missing = [skill.name for skill in selected if skill.required_context and not any(item in normalized for item in skill.required_context)]
    confidence = min(0.97, 0.62 + 0.12 * best_score + (0.08 if len(normalized) >= 6 else 0))
    if missing:
        return {"status": "clarify", "reason": "缺少关键上下文", "skills": [], "confidence": confidence, "question": f"为准确执行“{'、'.join(missing)}”，请补充目的地或地点信息。"}
    return {"status": "allow", "reason": "同义词与规则匹配通过", "skills": [skill.name for skill in selected], "confidence": confidence, "question": ""}
