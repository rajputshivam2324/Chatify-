from dataclasses import dataclass
from typing import Any

from huggingface_hub import InferenceClient
from pydantic import BaseModel

from app.core.config import get_settings

settings = get_settings()


class ModelConfig(BaseModel):
    model_id: str
    provider: str | None = None
    max_tokens: int
    temperature: float


class LLMInvocationError(Exception):
    """Raised when LLM invocation fails."""
    pass


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "llama": ModelConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        provider=None,
        max_tokens=10000,
        temperature=0.7,
    ),
    "qwen": ModelConfig(
        model_id="Qwen/Qwen2.5-VL-32B-Instruct",
        provider="fireworks-ai",
        max_tokens=25000,
        temperature=1.0,
    ),
    "gemma": ModelConfig(
        model_id="google/gemma-2-9b-it",
        provider=None,
        max_tokens=5000,
        temperature=0.7,
    ),
    "deepseek": ModelConfig(
        model_id="deepseek-ai/DeepSeek-R1",
        provider="sambanova",
        max_tokens=25000,
        temperature=1.0,
    ),
}

_client: InferenceClient | None = None


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        _client = InferenceClient(token=settings.hf_token)
    return _client


async def invoke_llm(model_key: str, messages: list[dict[str, Any]]) -> str:
    """
    Single entry point for ALL chat model calls.
    Never call InferenceClient directly outside this function.
    """
    if model_key not in MODEL_CONFIGS:
        raise LLMInvocationError(f"Unknown model: {model_key}")

    config = MODEL_CONFIGS[model_key]
    client = _get_client()

    try:
        kwargs: dict[str, Any] = {
            "model": config.model_id,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }

        if config.provider:
            kwargs["provider"] = config.provider

        response = client.chat_completion(**kwargs)

        if not response.choices or not response.choices[0].message:
            raise LLMInvocationError(f"Empty response from {model_key}")

        return response.choices[0].message.content

    except Exception as e:
        raise LLMInvocationError(f"LLM invocation failed for {model_key}: {str(e)}")