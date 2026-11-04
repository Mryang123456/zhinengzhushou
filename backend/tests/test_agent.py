import asyncio
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from backend.app.agent.service import checkpoint_config, checkpoint_scope, evaluate_expression, stream_events, weather_code_description
from langgraph.checkpoint.memory import MemorySaver
from backend.app.memory.store import SQLiteMemoryStore
from backend.app.models.factory import ModelConfigurationError, get_model_settings
from backend.app.security import SlidingWindowRateLimiter, valid_api_key


class CalculatorTests(unittest.TestCase):
    def test_evaluates_supported_arithmetic(self):
        self.assertEqual(evaluate_expression("(17 + 3) * 2"), 40)

    def test_rejects_code(self):
        with self.assertRaises(ValueError):
            evaluate_expression("__import__('os').system('whoami')")

    def test_translates_known_weather_code(self):
        self.assertEqual(weather_code_description(0), "晴朗")

    def test_stream_events_include_token_tool_and_done(self):
        class FakeGraph:
            async def astream(self, *_args, **_kwargs):
                yield "custom", {"type": "token", "content": "你好"}
                yield "updates", {"tools": {"messages": [SimpleNamespace(name="天气工具")]}}
                yield "updates", {"agent": {"messages": [SimpleNamespace(content="你好")]}}

        async def collect():
            return [event async for event in stream_events(FakeGraph(), "测试", "thread-1")]

        events = asyncio.run(collect())
        self.assertEqual([event["type"] for event in events], ["token", "tool_end", "done"])
        self.assertEqual(events[-1]["content"], "你好")

    def test_sqlite_memory_is_user_isolated_and_persistent(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "memory.db")
            memory_id = store.remember("张三", "我喜欢晴天出行。")
            self.assertEqual(store.recent("张三"), [(memory_id, "我喜欢晴天出行。")])
            self.assertEqual(store.recent("李四"), [])
            self.assertTrue(store.forget("张三", memory_id))

    def test_rejects_unknown_model_provider(self):
        previous = __import__("os").environ.get("MODEL_PROVIDER")
        try:
            __import__("os").environ["MODEL_PROVIDER"] = "unknown"
            with self.assertRaises(ModelConfigurationError):
                get_model_settings()
        finally:
            if previous is None:
                __import__("os").environ.pop("MODEL_PROVIDER", None)
            else:
                __import__("os").environ["MODEL_PROVIDER"] = previous

    def test_rate_limiter_rejects_excess_requests_until_window_expires(self):
        limiter = SlidingWindowRateLimiter(limit=2, window_seconds=10)
        self.assertTrue(limiter.allow("127.0.0.1", now=1))
        self.assertTrue(limiter.allow("127.0.0.1", now=2))
        self.assertFalse(limiter.allow("127.0.0.1", now=3))
        self.assertTrue(limiter.allow("127.0.0.1", now=12))

    def test_api_key_validation(self):
        self.assertTrue(valid_api_key(None, None))
        self.assertFalse(valid_api_key("错误", "正确"))
        self.assertTrue(valid_api_key("正确", "正确"))

    def test_checkpoint_key_is_user_isolated(self):
        self.assertEqual(
            checkpoint_config("会话-1", "张三")["configurable"]["thread_id"],
            "张三:会话-1",
        )
        self.assertNotEqual(checkpoint_config("会话-1", "张三"), checkpoint_config("会话-1", "李四"))

    def test_checkpoint_scope_uses_memory_saver_without_database(self):
        async def get_checkpointer():
            async with checkpoint_scope() as checkpointer:
                return checkpointer

        self.assertIsInstance(asyncio.run(get_checkpointer()), MemorySaver)


if __name__ == "__main__":
    unittest.main()
