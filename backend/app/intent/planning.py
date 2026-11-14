"""复杂问句的受控任务拆解。"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class IntentTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["weather", "web_search"]
    text: str = Field(min_length=2, max_length=250)
    location: str = Field(default="", max_length=80)
    date: str = Field(default="", max_length=32)
    count: int | None = Field(default=None, ge=1, le=8)
    condition: Literal["always", "if_rain"] = "always"
    depends_on: int | None = Field(default=None, ge=0, le=2)


class IntentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[IntentTask] = Field(max_length=3)


COMPOUND_MARKER = re.compile(r"如果|要是|若[^干]|的话|就推荐|再(?:查|搜|推荐|看看)|并且|同时|以及|然后")
RAIN_CONDITION = re.compile(r"(?:如果|要是|若)(?:会)?下雨|下雨(?:的?话|就)")
EXTERNAL_ACTION = re.compile(r"查|搜|推荐|找|预报")
CONDITIONAL_MARKER = re.compile(r"如果|要是|若[^干]|的话|才(?:查|搜|推荐)|否则|不然")
NEGATED_ACTION = re.compile(r"(?:不要|不用|无需|禁止|别|不)\s*(?:再|去|联网|上网)?\s*(?:查|搜|推荐|找)|(?:不|没)下雨|下雨.*(?:不|别)(?:查|搜|推荐|找)")


def planning_candidate(question: str) -> bool:
    return (
        len(question) <= 1000
        and bool(COMPOUND_MARKER.search(question) and EXTERNAL_ACTION.search(question))
        and (bool(re.search(r"天气|下雨|气温|降雨", question)) or len(EXTERNAL_ACTION.findall(question)) >= 2)
    )


def validate_plan(plan: IntentPlan, question: str) -> list[dict]:
    """只接受能回溯到用户原文的只读任务及向前依赖。"""
    tasks = plan.tasks
    if len(tasks) < 2 or len({task.text for task in tasks}) != len(tasks):
        return []
    rain_requested = bool(RAIN_CONDITION.search(question))
    if NEGATED_ACTION.search(question) or (CONDITIONAL_MARKER.search(question) and not rain_requested):
        return []
    if rain_requested and re.search(r"否则|不然|并且|同时|以及", question):
        return []
    if rain_requested != any(task.condition == "if_rain" for task in tasks):
        return []
    for index, task in enumerate(tasks):
        if task.text not in question or any(entity and entity not in question for entity in (task.location, task.date)):
            return []
        if task.depends_on is not None and task.depends_on >= index:
            return []
        if task.kind == "weather":
            if not task.location or task.condition != "always" or task.depends_on is not None:
                return []
            if not re.search(r"天气|气温|下雨|降雨|预报", task.text):
                return []
        elif rain_requested and task.condition != "if_rain":
            return []
        elif task.condition == "if_rain":
            if task.depends_on is None or tasks[task.depends_on].kind != "weather":
                return []
            if task.location and task.location != tasks[task.depends_on].location:
                return []
    return [task.model_dump() for task in tasks]


def forecast_has_rain(report: str) -> bool | None:
    if not report:
        return None
    if re.search(r"雷暴|雷雨|阵雨|小雨|中雨|大雨|暴雨|毛毛雨", report):
        return True
    if re.search(r"晴朗|多云|阴天|有雾|小雪|中雪|大雪|雾凇", report):
        return False
    return None
