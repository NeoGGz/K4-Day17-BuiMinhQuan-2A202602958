from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map provider names and aliases to normalized names.

    E.g.:
    - `anthorpic`, `claude` -> `anthropic`
    - `google`, `google-genai` -> `gemini`
    - `open-router`, `open_router` -> `openrouter`
    """
    cleaned = value.strip().lower().replace("-", "_")
    mapping = {
        "openai": "openai",
        "gpt": "openai",
        "custom": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "google_genai": "gemini",
        "anthropic": "anthropic",
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "ollama": "ollama",
        "openrouter": "openrouter",
        "open_router": "openrouter",
    }
    if cleaned in mapping:
        return mapping[cleaned]
    return cleaned


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Supported:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter` or OpenAI-compatible `ChatOpenAI`
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatOpenAI(**kwargs)

    elif provider == "custom":
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "base_url": config.base_url,
            "api_key": config.api_key or "EMPTY",
        }
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model_name": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    elif provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter

            kwargs = {"model": config.model_name, "temperature": config.temperature}
            if config.api_key:
                kwargs["api_key"] = config.api_key
            if config.base_url:
                kwargs["base_url"] = config.base_url
            return ChatOpenRouter(**kwargs)
        except ImportError:
            from langchain_openai import ChatOpenAI

            kwargs = {
                "model": config.model_name,
                "temperature": config.temperature,
                "base_url": config.base_url or "https://openrouter.ai/api/v1",
                "api_key": config.api_key or "EMPTY",
            }
            return ChatOpenAI(**kwargs)

    else:
        raise ValueError(f"Unsupported provider: {config.provider}")
