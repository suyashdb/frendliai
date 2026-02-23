"""
Gateway configuration.

Supports environment variables and model-specific reasoning detection profiles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Server-level settings loaded from env / .env file."""

    # Gateway
    host: str = "0.0.0.0"
    port: int = 8000

    # Upstream /chat/completions endpoint (the real LLM API)
    upstream_base_url: str = "http://localhost:9000/v1"
    upstream_api_key: str = "no-key"
    upstream_timeout_s: float = 120.0
    upstream_connect_timeout_s: float = 10.0
    # For FriendliAI dedicated endpoints: set this to your endpoint ID
    # (e.g. depjs3iq370hmqt). When set, the gateway appends it to the URL
    # path and omits the `model` field from the upstream request body.
    upstream_endpoint_id: str | None = None

    # Summariser – can point to a different (cheaper/faster) model
    summariser_base_url: str | None = None  # defaults to upstream_base_url
    summariser_api_key: str | None = None   # defaults to upstream_api_key
    summariser_model: str = "gpt-4o-mini"   # fast, cheap – good for summaries
    summariser_timeout_s: float = 30.0

    # Behaviour
    max_reasoning_tokens: int = 16_384
    enable_prompt_summary: bool = True
    enable_reasoning_summary: bool = True
    max_retries: int = 2

    # Logging
    log_level: str = "INFO"

    # CORS – configurable for production deployments
    cors_origins: str = "*"  # comma-separated list or "*"

    model_config = {"env_prefix": "GW_", "env_file": ".env", "extra": "ignore"}

    @property
    def resolved_summariser_base_url(self) -> str:
        return self.summariser_base_url or self.upstream_base_url

    @property
    def resolved_summariser_api_key(self) -> str:
        return self.summariser_api_key or self.upstream_api_key


# ---------------------------------------------------------------------------
# Model reasoning profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReasoningProfile:
    """Defines how to detect reasoning tokens for a given model family."""

    name: str
    # Regex that marks the START of reasoning block in the accumulated text
    start_pattern: re.Pattern[str]
    # Regex that marks the END of reasoning block
    end_pattern: re.Pattern[str]
    # Whether reasoning block is inclusive of the markers themselves
    strip_markers: bool = True
    # Some models emit reasoning as a separate field (e.g., `reasoning_content`)
    # If set, we look for this key in the delta instead of pattern-matching content.
    delta_field: str | None = None
    # How many chars to accumulate before concluding there's no reasoning.
    # Some models emit a preamble (acknowledgment, greeting) before <think>.
    # Set higher for models with longer preambles.
    no_reasoning_threshold: int = 500


# Pre-built profiles for popular reasoning models
REASONING_PROFILES: dict[str, ReasoningProfile] = {
    # DeepSeek-R1 family: uses <think>...</think>
    "deepseek-r1": ReasoningProfile(
        name="deepseek-r1",
        start_pattern=re.compile(r"<think>", re.IGNORECASE),
        end_pattern=re.compile(r"</think>", re.IGNORECASE),
        strip_markers=True,
    ),
    # Qwen QwQ family: also uses <think>...</think>
    "qwen-qwq": ReasoningProfile(
        name="qwen-qwq",
        start_pattern=re.compile(r"<think>", re.IGNORECASE),
        end_pattern=re.compile(r"</think>", re.IGNORECASE),
        strip_markers=True,
    ),
    # Generic think-tag models (catch-all for <think> pattern)
    "think-tag": ReasoningProfile(
        name="think-tag",
        start_pattern=re.compile(r"<think>", re.IGNORECASE),
        end_pattern=re.compile(r"</think>", re.IGNORECASE),
        strip_markers=True,
    ),
    # Models that use a separate `reasoning_content` delta field
    # (e.g., some OpenAI o-series via compatible APIs)
    "reasoning-field": ReasoningProfile(
        name="reasoning-field",
        start_pattern=re.compile(r"$^"),  # never matches – we use delta_field
        end_pattern=re.compile(r"$^"),
        strip_markers=False,
        delta_field="reasoning_content",
    ),
}

# Mapping from model name substrings → profile key.  Checked in order.
# FriendliAI serverless supports parse_reasoning via stream_options, which
# returns reasoning in a separate reasoning_content delta field.
MODEL_PROFILE_MAP: list[tuple[str, str]] = [
    ("deepseek-r1", "reasoning-field"),
    ("deepseek-reasoner", "reasoning-field"),
    ("qwq", "reasoning-field"),
    ("qwen3", "reasoning-field"),
    ("o1", "reasoning-field"),
    ("o3", "reasoning-field"),
    ("o4", "reasoning-field"),
]

# Fallback: if no model match, use think-tag as default
DEFAULT_PROFILE_KEY = "think-tag"


def get_reasoning_profile(model: str) -> ReasoningProfile:
    """Resolve the reasoning detection profile for a given model identifier."""
    model_lower = model.lower()
    for substring, profile_key in MODEL_PROFILE_MAP:
        if substring in model_lower:
            return REASONING_PROFILES[profile_key]
    return REASONING_PROFILES[DEFAULT_PROFILE_KEY]
