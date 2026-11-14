"""仅用同一会话最近一轮的天气上下文补全追问。"""

import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.app.intent.planning import RAIN_CONDITION
from backend.app.quality.validator import weather_tool_answer

DATE = r"今天|明天|后天|\d{4}-\d{2}-\d{2}|\d{1,2}月\d{1,2}[日号]"
DATE_FOLLOWUP = re.compile(rf"^(?:那|那么|改成|改到|换成)?\s*(?P<date>{DATE})\s*(?:呢|怎么样|可以吗)?[？?。]?$")
CITY_CORRECTION = re.compile(r"^(?:不对[，, ]*)?(?:改查|地点改成|城市换成|换个地方[，, ]*去)(?P<city>[\u4e00-\u9fff]{2,12})[？?。]?$")


def resolve_weather_followup(question: str, messages: list, previous_intent: dict) -> str:
    if not messages or not isinstance(messages[-1], HumanMessage):
        return question
    previous_index = next((i for i in range(len(messages) - 2, -1, -1) if isinstance(messages[i], HumanMessage)), None)
    if previous_index is None:
        return question
    date_change = DATE_FOLLOWUP.fullmatch(question.strip())
    city_change = CITY_CORRECTION.fullmatch(question.strip())
    conditional = bool(RAIN_CONDITION.search(question) and re.search(r"推荐|查|搜|找", question))
    if not (date_change or city_change or conditional):
        return question

    previous_turn = messages[previous_index:-1]
    if not isinstance(previous_turn[-1], AIMessage) or previous_turn[-1].tool_calls:
        return question
    previous_question = previous_intent.get("resolved_question") or previous_turn[0].content
    if not isinstance(previous_question, str):
        return question
    if previous_intent.get("weather_assessment"):
        previous_question += "（按天气判断）"
    calls = {
        call["id"]: call["args"]
        for message in previous_turn if isinstance(message, AIMessage)
        for call in message.tool_calls if call["name"] == "get_weather"
    }
    verified = {}
    for message in previous_turn:
        if isinstance(message, ToolMessage) and message.tool_call_id in calls:
            args = calls[message.tool_call_id]
            report = weather_tool_answer(previous_question, [message])
            location = args.get("location")
            if report and isinstance(location, str) and location in previous_question:
                dates = re.findall(r"天气预报（(\d{4}-\d{2}-\d{2})）", report)
                verified[location] = dates[0] if len(dates) == 1 else ""
    if len(verified) != 1:
        return question
    location, previous_date = next(iter(verified.items()))
    if date_change:
        prefix = f"{date_change['date']}{location}天气"
    elif city_change:
        prefix = f"{previous_date}{city_change['city']}天气"
    elif re.match(r"^(?:那[，, ]*)?(?:如果|若|要是)?(?:会)?下雨(?:的话|就)?[，, ]*(?:请|帮我)?推荐", question):
        return f"查{previous_date}{location}天气，{question}"
    else:
        return question
    return prefix + ("适合出游吗" if previous_intent.get("weather_assessment") else "怎么样")
