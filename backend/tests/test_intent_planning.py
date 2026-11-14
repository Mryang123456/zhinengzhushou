"""结构化意图、实体继承和条件执行回归。"""

import asyncio
from datetime import date, timedelta
import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessageChunk

from backend.app.agent.service import build_agent
from backend.app.agent.turns import stream_events
from backend.app.intent.planning import IntentPlan, validate_plan


QUESTION = "查明天杭州天气，下雨的话推荐两个室内景点"
TASKS = [
    {"kind": "weather", "text": "查明天杭州天气", "location": "杭州", "date": "明天"},
    {"kind": "web_search", "text": "推荐两个室内景点", "count": 2, "condition": "if_rain", "depends_on": 0},
]


class IntentPlanningTests(unittest.TestCase):
    def test_weather_followups_keep_context_and_recommend_only_supported_places(self):
        third = "如果下雨，推荐两个适合带孩子的室内景点"
        chapter = "杭州甲馆是适合亲子的室内景点。"
        weak = "杭州乙馆是当地旅游景点。"
        checked_dates = []
        after = (date.today() + timedelta(days=2)).isoformat()
        plan_question = f"查{after}杭州天气，{third}"
        tasks = [
            TASKS[0] | {"text": f"查{after}杭州天气", "date": after},
            TASKS[1] | {"text": "推荐两个适合带孩子的室内景点"},
        ]

        class Model:
            def bind_tools(self, tools):
                return self

            async def astream(self, messages):
                question = messages[-1].content
                if question == plan_question:
                    result = {"tasks": tasks}
                elif question.startswith("推荐两个适合带孩子"):
                    result = {"items": [
                        {"name": "甲馆", "source": 1, "quote": chapter},
                        {"name": "乙馆", "source": 2, "quote": weak},
                    ]}
                else:
                    yield AIMessageChunk(content="", tool_calls=[{
                        "name": "get_weather", "id": f"weather-{len(checked_dates)}",
                        "args": {"location": "杭州", "date": "后天" if question == "那后天呢？" else "明天"},
                    }])
                    return
                yield AIMessageChunk(content=json.dumps(result, ensure_ascii=False))

        def forecast(location, requested, days, get_json):
            checked_dates.append(requested)
            target = {"明天": (date.today() + timedelta(days=1)).isoformat(), "后天": after}.get(requested, requested)
            return f"{location}天气预报（{target}）：小雨；最高气温 20°C。"

        async def collect():
            graph = await build_agent(memory_db_path=None)
            return [
                [event async for event in stream_events(graph, question, "trip-followup")][-1]["content"]
                for question in ("明天杭州适合旅游吗", "那后天呢？", third)
            ]

        with (
            patch("backend.app.agent.service.create_chat_model", return_value=Model()),
            patch("backend.app.agent.tools.query_weather", side_effect=forecast),
            patch("backend.app.agent.tools.DDGS") as search,
            patch.dict("os.environ", {"AGENT_DEEPSEEK_FETCH_ENABLED": ""}),
        ):
            search.return_value.text.return_value = [
                {"href": "https://example.org/a", "body": chapter},
                {"href": "https://example.org/b", "body": weak},
            ]
            first, second, last = asyncio.run(collect())
        self.assertIn("不太适合长时间在户外", first)
        self.assertIn(after, second)
        self.assertIn("不太适合长时间在户外", second)
        self.assertEqual(checked_dates, ["明天", "后天", after])
        self.assertIn("甲馆", last)
        self.assertIn("暂未核实更多", last)
        self.assertNotIn("乙馆", last)

    def test_rejects_invented_entities_and_invalid_dependencies(self):
        for changes in ({"location": "上海"}, {"condition": "always"}, {"depends_on": 1}):
            with self.subTest(changes=changes):
                plan = IntentPlan.model_validate({"tasks": [TASKS[0], TASKS[1] | changes]})
                self.assertEqual(validate_plan(plan, QUESTION), [])

    def test_graph_runs_search_only_when_matching_forecast_indicates_rain(self):
        class Model:
            async def astream(self, messages):
                if messages[-1].content == QUESTION:
                    yield AIMessageChunk(content=json.dumps({"tasks": TASKS}, ensure_ascii=False))
                else:
                    yield AIMessageChunk(content=json.dumps({"items": [{
                        "name": "甲博物馆", "source": 1, "quote": "杭州的室内景点包括甲博物馆和乙展览馆。",
                    }]}, ensure_ascii=False))

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=QUESTION)
            return [event async for event in stream_events(graph, QUESTION, "compound-intent")]

        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        for city, condition, expected in (
            ("杭州", "小雨", "甲博物馆"),
            ("杭州", "晴朗", "没有执行条件推荐"),
            ("上海", "小雨", "暂时无法确认降雨条件"),
        ):
            with (
                self.subTest(city=city, condition=condition),
                patch("backend.app.agent.service.create_chat_model", return_value=Model()),
                patch("backend.app.agent.tools.query_weather", return_value=f"{city}天气预报（{tomorrow}）：{condition}；最高气温 20°C。"),
                patch("backend.app.agent.tools.DDGS") as search,
            ):
                search.return_value.text.return_value = [
                    {"href": "https://example.com/museum", "body": "杭州的室内景点包括甲博物馆和乙展览馆。"},
                ]
                events = asyncio.run(collect())
                self.assertIn(expected, events[-1]["content"])
                if expected == "甲博物馆":
                    search.return_value.text.assert_called_once_with("杭州 室内景点", max_results=5)
                else:
                    search.assert_not_called()
