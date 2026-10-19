"""The chat model, picked by LLM_PROVIDER. Both providers stream tokens, which the portal shows (and speaks) as they
arrive, so the first words of a reply come out long before the reply is finished."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .config import env, env_float, require


def chat_model() -> BaseChatModel:
    provider = env("LLM_PROVIDER", "openai").lower()
    timeout = env_float("LLM_TIMEOUT_S", 60.0)
    if provider == "openai":
        # Any OpenAI-compatible endpoint: OpenAI, Azure OpenAI (/openai/v1/, model = the deployment name),
        # OpenRouter, Groq, Gemini's OpenAI endpoint, Ollama…
        from langchain_openai import ChatOpenAI

        options: dict[str, object] = {}
        if (temperature := env_float("LLM_TEMPERATURE")) is not None:
            options["temperature"] = temperature
        if effort := env("OPENAI_REASONING_EFFORT"):  # reasoning models: "none" or "low" answer fastest
            options["reasoning_effort"] = effort
        return ChatOpenAI(
            model=require("OPENAI_MODEL"),
            api_key=require("OPENAI_API_KEY"),
            base_url=env("OPENAI_BASE_URL") or None,
            streaming=True,
            timeout=timeout,
            max_retries=2,
            **options,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # No temperature: current Claude models reject sampling parameters.
        return ChatAnthropic(
            model=require("ANTHROPIC_MODEL"),
            api_key=require("ANTHROPIC_API_KEY"),
            base_url=env("ANTHROPIC_BASE_URL") or None,
            max_tokens=int(env("ANTHROPIC_MAX_TOKENS", "4096")),
            streaming=True,
            timeout=timeout,
            max_retries=2,
        )
    raise RuntimeError(f"LLM_PROVIDER must be openai or anthropic, not {provider!r}.")
