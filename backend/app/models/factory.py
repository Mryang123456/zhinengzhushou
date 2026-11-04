"""统一大模型工厂：切换提供商只需调整环境变量，无需修改 Agent 工作流。"""

from __future__ import annotations

import os
from dataclasses import dataclass


class ModelConfigurationError(RuntimeError):
    """模型提供商、依赖或凭据配置不正确。"""


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    name: str
    temperature: float
    base_url: str | None


DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "ollama": "qwen3:8b",
    "anthropic": "claude-sonnet-4-5",
    "google": "gemini-2.5-flash",
}


def get_model_settings() -> ModelSettings:
    """从环境变量读取模型配置。"""
    provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ModelConfigurationError(
            f"不支持的 MODEL_PROVIDER：{provider}。可选：{', '.join(DEFAULT_MODELS)}。"
        )
    try:
        temperature = float(os.getenv("MODEL_TEMPERATURE", "0"))
    except ValueError as error:
        raise ModelConfigurationError("MODEL_TEMPERATURE 必须是数字。") from error
    if not 0 <= temperature <= 2:
        raise ModelConfigurationError("MODEL_TEMPERATURE 必须在 0 到 2 之间。")
    return ModelSettings(
        provider=provider,
        name=os.getenv("MODEL_NAME", DEFAULT_MODELS[provider]).strip(),
        temperature=temperature,
        base_url=os.getenv("MODEL_BASE_URL") or None,
    )


def ensure_model_ready(settings: ModelSettings | None = None) -> None:
    """在启动前检查当前提供商必要的凭据。Ollama 由本机服务管理，无需 API Key。"""
    settings = settings or get_model_settings()
    required_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    required_key = required_keys.get(settings.provider)
    if required_key and not os.getenv(required_key):
        raise ModelConfigurationError(f"当前模型提供商缺少 {required_key}。")


def create_chat_model(settings: ModelSettings | None = None):
    """创建符合 LangChain 接口的聊天模型，调用方再绑定工具。"""
    settings = settings or get_model_settings()
    ensure_model_ready(settings)
    common = {"model": settings.name, "temperature": settings.temperature}

    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        if settings.base_url:
            common["base_url"] = settings.base_url
        return ChatOpenAI(**common)

    if settings.provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as error:
            raise ModelConfigurationError(
                "请安装 Ollama 适配器：python -m pip install langchain-ollama"
            ) from error
        if settings.base_url:
            common["base_url"] = settings.base_url
        return ChatOllama(**common)

    if settings.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as error:
            raise ModelConfigurationError(
                "请安装 Anthropic 适配器：python -m pip install langchain-anthropic"
            ) from error
        return ChatAnthropic(**common)

    if settings.provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as error:
            raise ModelConfigurationError(
                "请安装 Google 适配器：python -m pip install langchain-google-genai"
            ) from error
        return ChatGoogleGenerativeAI(**common)

    raise ModelConfigurationError(f"未实现的模型提供商：{settings.provider}")
