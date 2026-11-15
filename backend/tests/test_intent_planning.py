"""结构化意图、实体继承和条件执行回归。"""

import asyncio
from datetime import date, timedelta
import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from backend.app.agent.service import build_agent
from backend.app.agent.turns import checkpoint_config, stream_events
from backend.app.intent.policy import classify
from backend.app.intent.planning import IntentPlan, validate_plan
from backend.app.skills.registry import load_skills
from backend.app.config import SKILLS_DIR


QUESTION = "查明天杭州天气，下雨的话推荐两个室内景点"
TASKS = [
    {"kind": "weather", "text": "查明天杭州天气", "location": "杭州", "date": "明天"},
    {"kind": "web_search", "text": "推荐两个室内景点", "count": 2, "condition": "if_rain", "depends_on": 0},
]


class IntentPlanningTests(unittest.TestCase):
    def test_open_ended_local_food_recommendations_require_public_sources(self):
        skills = load_skills(SKILLS_DIR)

        for question in ("查询 北京有什么美食", "推荐成都值得吃的小吃", "上海有哪些特色菜"):
            with self.subTest(question=question):
                decision = classify(question, skills)
                self.assertTrue(decision["requires_web_evidence"])
                self.assertTrue(decision["local_food_discovery"])
                self.assertEqual(decision["web_query"], question)

        for question in ("把这段美食介绍润色一下", "介绍一下北京烤鸭的历史"):
            with self.subTest(question=question):
                self.assertFalse(classify(question, skills).get("local_food_discovery", False))

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

    def test_each_weather_task_must_keep_its_own_city_and_date(self):
        question = "查明天上海天气，同时查后天广州气温"
        first = {"kind": "weather", "text": "查明天上海天气", "location": "上海", "date": "明天"}
        second = {"kind": "weather", "text": "查后天广州气温", "location": "广州", "date": "后天"}
        for tasks in ([first | {"location": "广州"}, second], [first | {"date": "后天"}, second],
                      [first | {"date": ""}, second], [second, first]):
            with self.subTest(tasks=tasks):
                self.assertEqual(validate_plan(IntentPlan.model_validate({"tasks": tasks}), question), [])

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
                    search.return_value.text.assert_called_once_with("杭州 室内景点", max_results=50)
                else:
                    search.assert_not_called()
                steps = [event["steps"] for event in events if event["type"] == "plan_progress"]
                self.assertEqual([step["status"] for step in steps[0]], ["pending", "pending"])
                self.assertEqual(steps[-1][1]["status"], {
                    "甲博物馆": "partial", "没有执行条件推荐": "skipped", "暂时无法确认降雨条件": "blocked",
                }[expected])

    def test_durable_plan_resumes_after_weather_checkpoint_without_repeating_weather(self):
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        tasks = [task.model_dump() for task in IntentPlan.model_validate({"tasks": TASKS}).tasks]
        line = "杭州的室内景点包括甲博物馆和乙展览馆。"

        class Model:
            async def astream(self, messages):
                yield AIMessageChunk(content=json.dumps({"items": [
                    {"name": "甲博物馆", "source": 1, "quote": line},
                    {"name": "乙展览馆", "source": 1, "quote": line},
                ]}, ensure_ascii=False))

        async def collect():
            saver = MemorySaver()
            config = checkpoint_config("plan-thread")
            initial_graph = await build_agent(memory_db_path=None, checkpointer=saver)
            await initial_graph.aupdate_state(config, {
                "messages": [
                    HumanMessage(QUESTION, id="web:message-1"),
                    AIMessage(content="", tool_calls=[{
                        "name": "get_weather", "id": "fixture-0", "args": {"location": "杭州", "date": "明天"},
                    }]),
                    ToolMessage(
                        content=f"杭州天气预报（{tomorrow}）：小雨；最高气温 20°C。",
                        name="get_weather", tool_call_id="fixture-0",
                    ),
                ],
                "tool_calls_used": 1, "durable_run_id": "run-1",
                "intent": {"tasks": tasks, "plan_id": "fixture"},
            }, as_node="tools")
            resumed_graph = await build_agent(memory_db_path=None, checkpointer=saver, recovering=True)
            return [event async for event in stream_events(
                resumed_graph, QUESTION, "plan-thread", message_id="message-1", run_id="run-1",
            )]

        with (
            patch.dict("os.environ", {"AGENT_DEEPSEEK_FETCH_ENABLED": ""}),
            patch("backend.app.agent.service.create_chat_model", return_value=Model()),
            patch("backend.app.agent.tools.query_weather") as weather,
            patch("backend.app.agent.tools.DDGS") as search,
        ):
            search.return_value.text.return_value = [{"href": "https://example.org/trip", "body": line}]
            events = asyncio.run(collect())
        weather.assert_not_called()
        search.return_value.text.assert_called_once()
        self.assertEqual(events[0]["type"], "plan_progress")
        self.assertEqual([step["status"] for step in events[0]["steps"]], ["complete", "pending"])
        final_plan = [event for event in events if event["type"] == "plan_progress"][-1]
        self.assertEqual([step["status"] for step in final_plan["steps"]], ["complete", "complete"])
        self.assertIn("https://example.org/trip", events[-1]["content"])

    def test_rain_plan_supplements_only_the_incomplete_search_and_combines_evidence(self):
        first = "杭州的室内景点包括甲博物馆。"
        second = "杭州的室内景点包括乙展览馆。"
        first_url, second_url = "https://example.com/museum", "https://example.org/gallery"
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        test = self
        reviewed = []

        class Model:
            async def ainvoke(self, messages):
                data = json.loads(messages[-1].content)
                test.assertEqual(data["问题"], "杭州 推荐两个室内景点")
                test.assertEqual(data["已尝试"], [{"query": "杭州 室内景点"}])
                test.assertNotIn("天气预报", messages[-1].content)
                reviewed.append(data)
                return AIMessage(content=json.dumps({
                    "action": "search", "query": "室内展览馆", "missing": "第二处室内景点",
                }, ensure_ascii=False))

            async def astream(self, messages):
                if messages[-1].content == QUESTION:
                    payload = {"tasks": TASKS}
                else:
                    has_second = second in messages[0].content
                    payload = {"items": [{"name": "甲博物馆", "source": 2 if has_second else 1, "quote": first}]}
                    if has_second:
                        payload["items"].append({"name": "乙展览馆", "source": 1, "quote": second})
                yield AIMessageChunk(content=json.dumps(payload, ensure_ascii=False))

        async def collect():
            graph = await build_agent(memory_db_path=None, initial_question=QUESTION)
            return [event async for event in stream_events(graph, QUESTION, "compound-iterative")]

        for max_calls in (2, 6):
            with (
                self.subTest(max_calls=max_calls),
                patch.dict("os.environ", {"AGENT_DEEPSEEK_FETCH_ENABLED": ""}),
                patch("backend.app.agent.service.load_governance", return_value={"maxToolCalls": max_calls}),
                patch("backend.app.agent.service.create_chat_model", return_value=Model()),
                patch("backend.app.agent.tools.query_weather", return_value=f"杭州天气预报（{tomorrow}）：小雨；最高气温 20°C。") as weather,
                patch("backend.app.agent.tools.DDGS") as search,
            ):
                search.return_value.text.side_effect = [[{"href": first_url, "body": first}], [{"href": second_url, "body": second}]]
                frames = asyncio.run(collect())
                weather.assert_called_once()
                self.assertEqual(search.return_value.text.call_count, 2 if max_calls == 6 else 1)
                self.assertIn(first_url, frames[-1]["content"])
                if max_calls == 6:
                    self.assertEqual(search.return_value.text.call_args.args[0], "杭州 室内景点 室内展览馆")
                    self.assertIn(second_url, frames[-1]["content"])
                    self.assertEqual(sum(event["type"] == "evidence_retry" for event in frames), 1)
                else:
                    self.assertIn("暂未核实更多", frames[-1]["content"])
                    quality = next(event for event in frames if event["type"] == "quality")
                    self.assertTrue(quality["passed"] and quality["partial"])
        self.assertEqual(len(reviewed), 1)
