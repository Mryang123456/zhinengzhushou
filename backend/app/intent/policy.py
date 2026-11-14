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
CASUAL_REPLIES = {
    "你好": "你好！今天想聊些什么？",
    "嗨": "嗨！有什么想一起看看的吗？",
    "谢谢": "不客气！",
}
MEMORY_PATTERN = re.compile(r"记住|回忆|想起|删除.*记忆|忘记")
VERIFY_FOLLOWUP = re.compile(
    r"^(?:那(?:就)?|好(?:的)?[，,]?)?(?:请)?(?:帮我)?(?:联网|上网)?"
    r"(?:查证|核实|求证|验证)(?:一下)?(?:这(?:件事|个说法|条消息)|刚才(?:那个|的说法))?(?:吧|好吗)?[。！!?？]*$"
)
WEATHER_ACTIVITY = re.compile(r"旅游|旅行|出游|出行|出去玩|户外|徒步|爬山|登山|露营|骑行|野餐|逛景点")
SUITABILITY = re.compile(r"适合|适宜|能不能|可不可以|是否可以|可以.{0,20}吗|建议.{0,20}吗|还能.{0,20}吗")
WEATHER_CLOTHING = re.compile(r"穿(?:点|上|着)?什么|怎么穿|穿衣|带(?:件)?什么衣服|要不要穿")
LOCAL_CLOTHING_QUESTION = re.compile(
    r"(?:[一-鿿]{2,10}(?:现在|今天|明天|后天)|(?:现在|今天|明天|后天)[一-鿿]{0,10}?)"
    r"(?:适合|应该|出门)?(?:穿什么(?:样的)?(?:衣服)?(?:合适)?|怎么穿|怎么穿衣服)[？?。]?"
)
GIVEN_TEMPERATURE = re.compile(r"-?\d+(?:\.\d+)?\s*(?:℃|°\s*[cC]|度)")
PUBLIC_EVENT_QUESTION = re.compile(
    r"^(?P<subject>.{2,80}?)(?:为什么(?:会|要|突然)?|为何|是不是(?:已经)?|是否(?:已经)?|真的(?:已经)?|"
    r"已经|突然|刚刚|有没有)(?P<event>辞职|离职|卸任|下台|裁员|停产|倒闭|破产|退市|被收购|被撤职|取消|下架)"
    r"(?:了)?(?:吗|么|了吗|的原因是什么)?[？?。！!]*$"
)
GENERIC_EVENT_SUBJECT = re.compile(r"(?:人(?:们|类)?|员工|公司|企业|老板|高管|管理者|产品|工厂|(?:一个|一家|某些|有些|部分|某|哪些|什么样的).+)")
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


def casual_reply(question: str) -> str | None:
    return CASUAL_REPLIES.get(re.sub(r"\s+", "", question).rstrip("！!？?。"))


def is_verification_followup(question: str) -> bool:
    return bool(VERIFY_FOLLOWUP.fullmatch(re.sub(r"\s+", "", question)))


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
    if is_verification_followup(question):
        return {
            "status": "clarify", "reason": "缺少明确的查证对象", "skills": [], "confidence": 0.95,
            "question": "你想查证哪件事或哪条说法？发给我具体内容，也可以附上你看到的链接。",
        }
    greeting = casual_reply(question)
    if greeting:
        return {"status": "allow", "reason": "简短问候", "skills": [], "confidence": 0.95,
                "question": "", "direct_reply": greeting}
    if MATH_PATTERN.search(normalized) or CASUAL_PATTERN.match(normalized) or MEMORY_PATTERN.search(normalized):
        return {"status": "allow", "reason": "基础任务直通", "skills": [], "confidence": 0.95, "question": ""}
    implicit_weather = re.fullmatch(r"(?:今天|明天|后天)[\u4e00-\u9fff]{2,12}(?:适合|适宜)(?:旅游|旅行|出游|出去玩|徒步|露营)吗[？?。]?", normalized)
    activity = bool(WEATHER_ACTIVITY.search(normalized) and SUITABILITY.search(normalized))
    clothing = bool(WEATHER_CLOTHING.search(normalized)) and not GIVEN_TEMPERATURE.search(normalized)
    if (activity or clothing) and (
        implicit_weather or LOCAL_CLOTHING_QUESTION.fullmatch(normalized) or re.search(r"天气|气温|下雨|降雨", normalized)
    ):
        return {
            "status": "allow", "reason": "需要根据天气判断活动是否合适" if activity else "需要根据天气给出穿衣建议", "skills": [],
            "confidence": 0.95, "question": "", "weather_assessment": True,
            "weather_activity": activity, "weather_clothing": clothing,
        }
    event = PUBLIC_EVENT_QUESTION.fullmatch(normalized)
    if event:
        subject = re.sub(r"^(?:请问|请帮我查一下|帮我查一下|帮我查|请核实)", "", event["subject"])
        if subject and not GENERIC_EVENT_SUBJECT.fullmatch(subject):
            return {
                "status": "allow", "reason": "具体事件需要联网核实", "skills": [], "confidence": 0.95,
                "question": "", "requires_web_evidence": True, "web_query": f"{subject} {event['event']} 官方公告 核实",
            }
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
