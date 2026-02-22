"""
Stream processor – the core orchestration engine.

Coordinates:
  1. Parallel prompt summarisation (starts immediately for low TTFT)
  2. Upstream streaming + reasoning detection
  3. Reasoning summarisation (after reasoning block ends)
  4. Final output forwarding

All output is yielded as raw SSE bytes for the response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncIterator

import httpx

from gateway.config import Settings, ReasoningProfile, get_reasoning_profile
from gateway.models import (
    GatewayRequest,
    SSEChunk,
    ChunkChoice,
    DeltaContent,
    PhaseEvent,
)
from gateway.reasoning import StreamingReasoningDetector, DeltaFieldDetector
from gateway.summarizer import Summariser

logger = logging.getLogger(__name__)


def _make_chunk(
    completion_id: str,
    model: str,
    content: str,
    role: str | None = None,
    finish_reason: str | None = None,
) -> SSEChunk:
    return SSEChunk(
        id=completion_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChunkChoice(
                delta=DeltaContent(role=role, content=content),
                finish_reason=finish_reason,
            )
        ],
    )


async def process_stream(
    request: GatewayRequest,
    settings: Settings,
    http_client: httpx.AsyncClient,
    cached_prompt_summary: list[str] | None = None,
) -> AsyncIterator[bytes]:
    """
    Main entry point.  Yields raw SSE bytes (ready to write to response).

    If cached_prompt_summary is provided (from a pre-warmed session),
    Phase 1 uses the cache instead of calling the summariser — near-zero
    TTFT for the prompt summary phase.
    """
    completion_id = f"chatcmpl-gw-{uuid.uuid4().hex[:12]}"
    model = request.model

    # Resolve reasoning profile
    if request.reasoning_profile:
        from gateway.config import REASONING_PROFILES
        profile = REASONING_PROFILES.get(
            request.reasoning_profile, get_reasoning_profile(model)
        )
    else:
        profile = get_reasoning_profile(model)

    summariser = Summariser(settings)
    use_delta_field = profile.delta_field is not None

    # ------------------------------------------------------------------
    # PHASE 1: Prompt summary (runs in parallel with upstream call)
    # ------------------------------------------------------------------
    prompt_summary_task = None
    has_cached_summary = cached_prompt_summary is not None and len(cached_prompt_summary) > 0

    if not has_cached_summary and request.include_prompt_summary and settings.enable_prompt_summary:
        # No cache hit — compute prompt summary in parallel (standard path)
        prompt_messages = [m.model_dump(exclude_none=True) for m in request.messages]

        async def _collect_prompt_summary() -> list[str]:
            tokens: list[str] = []
            try:
                async for t in summariser.summarise_prompt(prompt_messages, http_client):
                    tokens.append(t)
            except Exception as exc:
                logger.warning("Prompt summarisation failed: %s", exc)
            return tokens

        prompt_summary_task = asyncio.create_task(_collect_prompt_summary())

    # ------------------------------------------------------------------
    # PHASE 2: Stream from upstream + detect reasoning
    # ------------------------------------------------------------------
    upstream_url = f"{settings.upstream_base_url.rstrip('/')}/chat/completions"
    upstream_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.upstream_api_key}",
    }
    upstream_body = request.to_upstream_body()

    # We'll collect reasoning and output separately
    reasoning_tokens: list[str] = []
    output_tokens: list[str] = []
    detector = (
        DeltaFieldDetector(profile.delta_field)
        if use_delta_field
        else StreamingReasoningDetector(profile)
    )
    upstream_error: Exception | None = None

    try:
        async with http_client.stream(
            "POST",
            upstream_url,
            json=upstream_body,
            headers=upstream_headers,
            timeout=httpx.Timeout(
                settings.upstream_timeout_s,
                connect=settings.upstream_connect_timeout_s,
            ),
        ) as upstream_resp:
            if upstream_resp.status_code != 200:
                error_body = await upstream_resp.aread()
                logger.error(
                    "Upstream returned %d: %s",
                    upstream_resp.status_code,
                    error_body[:1000],
                )
                # Yield an error and bail
                error_msg = f"Upstream error: {upstream_resp.status_code}"
                yield _make_chunk(completion_id, model, error_msg).to_sse_bytes()
                yield b"data: [DONE]\n\n"
                return

            async for line in upstream_resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue

                if use_delta_field:
                    assert isinstance(detector, DeltaFieldDetector)
                    cls, text = detector.classify_delta(delta)
                    if cls == "reasoning":
                        reasoning_tokens.append(text)
                    elif cls == "output":
                        output_tokens.append(text)
                else:
                    assert isinstance(detector, StreamingReasoningDetector)
                    content = delta.get("content", "")
                    if not content:
                        continue
                    events = detector.feed(content)
                    for cls, text in events:
                        if cls == "output":
                            output_tokens.append(text)
                        elif cls == "reasoning_complete":
                            reasoning_tokens.append(text)
                        elif cls == "buffered":
                            pass  # held internally

    except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
        upstream_error = exc
        logger.error("Upstream connection failed: %s", exc)
        yield _make_chunk(
            completion_id, model, f"[Gateway error: upstream unavailable – {type(exc).__name__}]"
        ).to_sse_bytes()
        yield b"data: [DONE]\n\n"
        return

    # Finalize detector – flush any buffered content
    if isinstance(detector, StreamingReasoningDetector):
        result = detector.finalize()
        if not reasoning_tokens and result.reasoning_text:
            reasoning_tokens = [result.reasoning_text]
        if not output_tokens and result.output_tokens:
            output_tokens = result.output_tokens

    # ------------------------------------------------------------------
    # Now yield everything to the client in order
    # ------------------------------------------------------------------

    # --- Phase 1: Prompt Summary ---
    if has_cached_summary or prompt_summary_task is not None:
        yield PhaseEvent(phase="prompt_summary", label="Prompt Summary").to_sse_bytes()
        # Role chunk
        yield _make_chunk(completion_id, model, "", role="assistant").to_sse_bytes()

        if has_cached_summary:
            # Serve from pre-warmed session cache — near-zero latency
            logger.info("Serving prompt summary from warm session cache")
            for t in cached_prompt_summary:
                yield _make_chunk(completion_id, model, t).to_sse_bytes()
        else:
            try:
                summary_tokens = await asyncio.wait_for(prompt_summary_task, timeout=15.0)
                for t in summary_tokens:
                    yield _make_chunk(completion_id, model, t).to_sse_bytes()
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning("Prompt summary timed out or failed: %s", exc)
                yield _make_chunk(
                    completion_id, model, "[Prompt summary unavailable]"
                ).to_sse_bytes()

    # --- Phase 2: Reasoning Summary ---
    reasoning_text = "".join(reasoning_tokens)
    if (
        reasoning_text.strip()
        and request.include_reasoning_summary
        and settings.enable_reasoning_summary
    ):
        yield PhaseEvent(
            phase="reasoning_summary", label="Reasoning Summary"
        ).to_sse_bytes()
        try:
            async for t in summariser.summarise_reasoning(reasoning_text, http_client):
                yield _make_chunk(completion_id, model, t).to_sse_bytes()
        except Exception as exc:
            logger.warning("Reasoning summarisation failed: %s", exc)
            yield _make_chunk(
                completion_id, model, "[Reasoning summary unavailable]"
            ).to_sse_bytes()

    # --- Phase 3: Final output ---
    yield PhaseEvent(phase="output", label="Response").to_sse_bytes()
    for token in output_tokens:
        yield _make_chunk(completion_id, model, token).to_sse_bytes()

    # Finish
    yield _make_chunk(
        completion_id, model, "", finish_reason="stop"
    ).to_sse_bytes()
    yield b"data: [DONE]\n\n"
