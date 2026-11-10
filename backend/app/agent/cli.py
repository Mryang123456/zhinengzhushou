"""中文交互式 CLI 及单次命令入口。"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from backend.app.agent.service import build_agent
from backend.app.agent.turns import ask, checkpoint_scope, stream_events
from backend.app.config import MCP_CONFIG_FILE, MEMORY_DB_FILE
from backend.app.models.factory import ModelConfigurationError, ensure_model_ready, get_model_settings


def is_billing_error(error: Exception) -> bool:
    """识别 OpenAI 的常见计费或额度错误文案。"""
    message = str(error).lower()
    return any(
        phrase in message
        for phrase in (
            "account is not active",
            "no credits remaining",
            "insufficient_quota",
            "credit_balance_exhausted",
        )
    )


async def answer_or_exit(app, question: str, thread_id: str, user_id: str = "default") -> str:
    """将常见账户配置失败转换为可执行的命令行提示。"""
    try:
        return await ask(app, question, thread_id, user_id)
    except Exception as error:
        if is_billing_error(error):
            raise SystemExit(
                "OpenAI API 项目尚未启用。请先在 OpenAI Platform 启用 API 计费后重试。"
            ) from error
        raise


async def run_cli(args: argparse.Namespace) -> None:
    mcp_config_path = None if args.no_mcp else Path(args.mcp_config)
    memory_db_path = None if args.no_memory else Path(args.memory_db)
    checkpoint_database_url = args.checkpoint_database_url or os.getenv("CHECKPOINT_DATABASE_URL")
    async with checkpoint_scope(checkpoint_database_url) as checkpointer:
        app = await build_agent(
            mcp_config_path,
            args.user_id,
            memory_db_path,
            args.memory_database_url or os.getenv("MEMORY_DATABASE_URL"),
            checkpointer,
        )
        if args.once:
            if args.no_stream:
                print(await answer_or_exit(app, args.once, args.thread_id, args.user_id))
            else:
                await print_stream(app, args.once, args.thread_id, args.user_id)
            return

        print("智能体已就绪。输入 /quit 退出。")
        while True:
            question = input("> ").strip()
            if question in {"/quit", "/exit"}:
                return
            if question:
                if args.no_stream:
                    print(await answer_or_exit(app, question, args.thread_id, args.user_id))
                else:
                    await print_stream(app, question, args.thread_id, args.user_id)


async def print_stream(app, question: str, thread_id: str, user_id: str = "default") -> None:
    """将流式事件渲染为中文命令行输出。"""
    printed_token = False
    try:
        async for event in stream_events(app, question, thread_id, user_id):
            if event["type"] == "token":
                print(event["content"], end="", flush=True)
                printed_token = True
            elif event["type"] == "tool_start":
                print(f"\n[正在调用工具：{'、'.join(event['tools'])}]", flush=True)
            elif event["type"] == "tool_end":
                print(f"\n[工具调用完成：{'、'.join(event['tools'])}]", flush=True)
            elif event["type"] == "done":
                if not printed_token:
                    print(event["content"], end="", flush=True)
                print()
    except Exception as error:
        if is_billing_error(error):
            raise SystemExit(
                "OpenAI API 项目尚未启用。请先在 OpenAI Platform 启用 API 计费后重试。"
            ) from error
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="基于 LangChain 和 LangGraph 的基础智能体")
    parser.add_argument("--once", help="执行一个问题后退出")
    parser.add_argument("--thread-id", default="demo", help="对话检查点标识")
    parser.add_argument(
        "--mcp-config",
        default=str(MCP_CONFIG_FILE),
        help="显式 MCP 服务配置文件的路径",
    )
    parser.add_argument("--no-mcp", action="store_true", help="仅启用内置工具")
    parser.add_argument("--no-stream", action="store_true", help="关闭命令行流式输出")
    parser.add_argument("--user-id", default="default", help="长期记忆所属用户标识")
    parser.add_argument(
        "--memory-db",
        default=str(MEMORY_DB_FILE),
        help="SQLite 长期记忆数据库路径",
    )
    parser.add_argument("--no-memory", action="store_true", help="关闭长期记忆功能")
    parser.add_argument(
        "--memory-database-url",
        help="PostgreSQL 长期记忆数据库地址；优先级高于本地 SQLite",
    )
    parser.add_argument(
        "--checkpoint-database-url",
        help="PostgreSQL 检查点数据库地址；用于跨重启恢复任务状态",
    )
    args = parser.parse_args()

    try:
        ensure_model_ready(get_model_settings())
    except ModelConfigurationError as error:
        raise SystemExit(f"模型配置错误：{error}") from error

    asyncio.run(run_cli(args))


if __name__ == "__main__":
    main()
