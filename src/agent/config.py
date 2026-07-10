"""Shared configuration for all agents."""

import logging
import os
from dataclasses import dataclass

import dotenv
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable, RunnableLambda

from src.middleware.retry_middleware import (
    RETRYABLE_FINISH_REASONS,
    MalformedResponseError,
    ModelRetryMiddleware,
)
from src.middleware.tool_retry_middleware import ToolRetryMiddleware

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# Model Registry
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for a supported chat model."""

    id: str  # e.g., "google_genai:gemini-3.1-flash-lite"
    name: str  # Display name, e.g., "Gemini 3.1 Flash Lite"
    provider: str  # e.g., "google", "openai", "baseten"
    api_key_env: str  # Environment variable for API key
    description: str | None = None
    base_url: str | None = None


# Backend-supported models.
MODELS: dict[str, ModelConfig] = {
    # Anthropic
    "claude-haiku-4.5": ModelConfig(
        id="anthropic:claude-haiku-4-5-20251001",
        name="Claude Haiku 4.5",
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        description="Fast and cheap Anthropic model",
    ),
    # OpenAI
    "gpt-5.4-nano": ModelConfig(
        id="openai:gpt-5.4-nano",
        name="GPT-5.4 Nano",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        description="Cheapest GPT-5.4-class model for simple high-volume tasks",
    ),
    # DeepSeek OpenAI-compatible API
    "deepseek-v4-flash": ModelConfig(
        id="openai:deepseek-v4-flash",
        name="DeepSeek V4 Flash",
        provider="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        description="Fast DeepSeek model for agent and retrieval workflows",
        base_url="https://api.deepseek.com",
    ),
    "deepseek-v4-pro": ModelConfig(
        id="openai:deepseek-v4-pro",
        name="DeepSeek V4 Pro",
        provider="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        description="Stronger DeepSeek model for higher-quality synthesis",
        base_url="https://api.deepseek.com",
    ),
    # Google
    "gemini-3.1-flash-lite": ModelConfig(
        id="google_genai:gemini-3.1-flash-lite",
        name="Gemini 3.1 Flash Lite",
        provider="google",
        api_key_env="GOOGLE_API_KEY",
        description="Fastest, most cost-effective Gemini",
    ),
}

def _model_from_env(env_name: str, default_key: str) -> ModelConfig:
    """Select a model config by environment variable with validation."""
    model_key = os.getenv(env_name, default_key)
    if model_key not in MODELS:
        available = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown {env_name}={model_key!r}. Available: {available}")
    return MODELS[model_key]


def _fallback_models_from_env() -> list[ModelConfig]:
    """Return fallback models that have API keys configured."""
    raw_keys = os.getenv("FALLBACK_MODEL_KEYS", "gpt-5.4-nano,claude-haiku-4.5")
    fallback_keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
    fallback_models = []
    for key in fallback_keys:
        if key not in MODELS:
            logger.warning("Ignoring unknown fallback model key: %s", key)
            continue
        model = MODELS[key]
        if os.getenv(model.api_key_env):
            fallback_models.append(model)
    return fallback_models


# Default models for different use cases
DEFAULT_MODEL = _model_from_env("DEFAULT_MODEL_KEY", "deepseek-v4-flash")
GUARDRAILS_MODEL = _model_from_env("GUARDRAILS_MODEL_KEY", "gpt-5.4-nano")

# Fallback chain (only models with configured API keys are enabled)
FALLBACK_MODELS = _fallback_models_from_env()

# =============================================================================
# API Key Setup
# =============================================================================

API_KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "DEEPSEEK_API_KEY",
]

for key in API_KEYS:
    if value := os.getenv(key):
        os.environ[key] = value.strip()
        logger.info(f"{key} configured")


# =============================================================================
# Model Initialization
# =============================================================================

# Retry configuration
MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "2"))


def _init_chat_model_from_config(model_config: ModelConfig) -> Runnable:
    """Initialize a chat model, including OpenAI-compatible providers."""
    kwargs = {}
    if model_config.base_url:
        kwargs["base_url"] = model_config.base_url
    if api_key := os.getenv(model_config.api_key_env):
        kwargs["api_key"] = api_key
    return init_chat_model(model=model_config.id, **kwargs)


# Primary model. Public callers cannot switch this at runtime.
if not os.getenv(DEFAULT_MODEL.api_key_env):
    raise ValueError(
        f"{DEFAULT_MODEL.api_key_env} is required for DEFAULT_MODEL_KEY="
        f"{os.getenv('DEFAULT_MODEL_KEY', 'deepseek-v4-flash')!r}"
    )
default_model = _init_chat_model_from_config(DEFAULT_MODEL)
logger.info(f"Default model: {DEFAULT_MODEL.name} ({DEFAULT_MODEL.id})")


def _raise_for_retryable_finish_reason(response: object) -> object:
    metadata = getattr(response, "response_metadata", None) or {}
    finish_reason = metadata.get("finish_reason", "")
    if finish_reason in RETRYABLE_FINISH_REASONS:
        raise MalformedResponseError(f"Model returned {finish_reason}")
    return response


def _init_retrying_model(model: ModelConfig) -> Runnable:
    return (
        _init_chat_model_from_config(model) | RunnableLambda(_raise_for_retryable_finish_reason)
    ).with_retry(stop_after_attempt=MAX_RETRIES + 1)


def init_retry_fallback_model(model: ModelConfig) -> Runnable:
    """Initialize a model runnable with the shared retry and fallback policy."""
    primary_model = _init_retrying_model(model)
    fallback_models = [_init_retrying_model(fallback) for fallback in FALLBACK_MODELS]
    return primary_model.with_fallbacks(fallback_models)


summarization_model = init_retry_fallback_model(DEFAULT_MODEL)

# =============================================================================
# Middleware
# =============================================================================

model_retry_middleware = ModelRetryMiddleware(max_retries=MAX_RETRIES)
tool_retry_middleware = ToolRetryMiddleware(max_attempts=3)

model_fallback_middleware = (
    ModelFallbackMiddleware(*[m.id for m in FALLBACK_MODELS])
    if FALLBACK_MODELS
    else None
)
fallback_names = " -> ".join(m.name for m in FALLBACK_MODELS) or "none"
logger.info(f"Fallback chain: {fallback_names}")

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Models
    "MODELS",
    "DEFAULT_MODEL",
    "GUARDRAILS_MODEL",
    "FALLBACK_MODELS",
    "ModelConfig",
    # Models
    "default_model",
    "init_retry_fallback_model",
    "summarization_model",
    # Middleware
    "model_retry_middleware",
    "tool_retry_middleware",
    "model_fallback_middleware",
    # Config
    "MAX_RETRIES",
    "logger",
]
