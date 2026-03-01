"""
Request / response models – OpenAI-compatible with gateway extensions.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Inbound request (OpenAI-compatible + gateway extras)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class GatewayRequest(BaseModel):
    """
    Mirrors the OpenAI /chat/completions request body.
    Adds optional gateway-specific fields.
    """
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = True  # gateway always streams; ignored if False
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None

    # --- Gateway extensions ---
    # Allow caller to disable summaries per-request
    include_prompt_summary: bool = True
    include_reasoning_summary: bool = True
    # Override reasoning profile (e.g., "deepseek-r1", "think-tag")
    reasoning_profile: str | None = None
    # Attach to a pre-warmed session (from POST /v1/sessions/warm)
    session_id: str | None = None

    model_config = {"extra": "allow"}

    def to_upstream_body(self, include_model: bool = True) -> dict[str, Any]:
        """Build the body forwarded to the upstream /chat/completions API."""
        body: dict[str, Any] = {
            "messages": [m.model_dump(exclude_none=True) for m in self.messages],
            "stream": True,  # always stream from upstream
        }
        # Only include model field for serverless endpoints.
        # Dedicated endpoints identify the model via the URL path.
        if include_model:
            body["model"] = self.model
        # Standard optional fields
        for opt in (
            "temperature", "top_p", "max_tokens", "stop",
            "presence_penalty", "frequency_penalty", "user",
        ):
            val = getattr(self, opt)
            if val is not None:
                body[opt] = val
        # Forward any extra fields (tools, response_format, seed, logprobs, etc.)
        gateway_only = {
            "include_prompt_summary", "include_reasoning_summary",
            "reasoning_profile", "session_id", "stream",
        }
        if self.model_extra:
            for key, val in self.model_extra.items():
                if key not in gateway_only:
                    body[key] = val
        return body


# ---------------------------------------------------------------------------
# SSE chunk shapes (outbound to client)
# ---------------------------------------------------------------------------

class DeltaContent(BaseModel):
    role: str | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaContent
    finish_reason: str | None = None


class SSEChunk(BaseModel):
    """One SSE data payload – OpenAI-shaped."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]

    def to_sse_bytes(self) -> bytes:
        return f"data: {self.model_dump_json()}\n\n".encode()


class PhaseEvent(BaseModel):
    """Custom SSE event marking a new output phase."""
    phase: Literal["prompt_summary", "reasoning_summary", "output"]
    label: str
    step: int | None = None  # Step number for incremental reasoning summaries

    def to_sse_bytes(self) -> bytes:
        return f"event: phase\ndata: {self.model_dump_json()}\n\n".encode()


# ---------------------------------------------------------------------------
# Session warmup models
# ---------------------------------------------------------------------------

class WarmupRequest(BaseModel):
    """Request to pre-warm a session for faster subsequent completions."""
    model: str
    system_prompt: str | None = None
    messages: list[ChatMessage] | None = None

class WarmupResponse(BaseModel):
    """Response confirming session creation with warmup status."""
    session_id: str
    model: str
    status: str = "warming"
    prompt_summary_ready: bool = False
    kv_cache_warmed: bool = False
    ttl_seconds: int = 300
