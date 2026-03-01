"""
Stream processor – interleaved reasoning summarisation pipeline.

Architecture:
  - Producer task reads upstream SSE, detects reasoning steps, pushes events to queue
  - Consumer (async generator) reads queue, summarises each step inline, yields SSE bytes

This enables incremental reasoning summaries — the client sees reasoning insights
streaming in real-time while the model is still thinking, rather than waiting for
the entire reasoning block to complete.

Pipeline:
  Phase 1: Prompt summary (from cache or parallel task)
  Phase 2: Reasoning summaries (N steps, each summarised as detected)
  Phase 3: Final output (buffered during reasoning, forwarded after)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Any

import httpx

from gateway.config import Settings, ReasoningProfile, get_reasoning_profile
from gateway.models import (
    GatewayRequest,
    SSEChunk,
    ChunkChoice,
    DeltaContent,
    PhaseEvent,
)
from gateway.reasoning import (
    StreamingReasoningDetector,
    DeltaFieldDetector,
    ReasoningStep,
)
from gateway.summarizer import Summariser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal event types for the queue
# ---------------------------------------------------------------------------

@dataclass
class EvReasoningStep:
    step: ReasoningStep

@dataclass
class EvReasoningEnded:
    pass

@dataclass
class EvOutput:
    text: str

@dataclass
class EvUpstreamDone:
    pass

@dataclass
class EvError:
    message: str


# Type alias for queue events
StreamEvent = EvReasoningStep | EvReasoningEnded | EvOutput | EvUpstreamDone | EvError


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


# ---------------------------------------------------------------------------
# Producer: reads upstream, feeds detector, pushes events to queue
# ---------------------------------------------------------------------------

async def _upstream_producer(
    queue: asyncio.Queue[StreamEvent],
    upstream_url: str,
    upstream_headers: dict[str, str],
    upstream_body: dict,
    settings: Settings,
    http_client: httpx.AsyncClient,
    profile: ReasoningProfile,
) -> None:
    """
    Background task that:
      1. Streams from the upstream /chat/completions API
      2. Feeds tokens into the reasoning detector
      3. Pushes typed events (reasoning_step, output, etc.) to the queue
    """
    use_delta_field = profile.delta_field is not None
    detector = (
        DeltaFieldDetector(profile.delta_field)
        if use_delta_field
        else StreamingReasoningDetector(profile)
    )

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
        ) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                logger.error("Upstream returned %d: %s", resp.status_code, error_body[:1000])
                await queue.put(EvError(message=f"Upstream error: {resp.status_code}"))
                await queue.put(EvUpstreamDone())
                return

            async for line in resp.aiter_lines():
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
                    events = detector.classify_delta(delta)
                    for ev_type, *ev_data in events:
                        if ev_type == "reasoning_step":
                            await queue.put(EvReasoningStep(step=ev_data[0]))
                        elif ev_type == "output":
                            await queue.put(EvOutput(text=ev_data[0]))
                        elif ev_type == "reasoning_ended":
                            await queue.put(EvReasoningEnded())
                else:
                    assert isinstance(detector, StreamingReasoningDetector)
                    content = delta.get("content", "")
                    if not content:
                        continue
                    events = detector.feed(content)
                    for ev_type, *ev_data in events:
                        logger.debug("detector event: %s state=%s", ev_type, detector.state)
                        if ev_type == "reasoning_step":
                            logger.info("reasoning_step #%d len=%d", ev_data[0].index, len(ev_data[0].text))
                            await queue.put(EvReasoningStep(step=ev_data[0]))
                        elif ev_type == "output":
                            await queue.put(EvOutput(text=ev_data[0]))
                        elif ev_type == "reasoning_ended":
                            logger.info("reasoning_ended, steps=%d", detector.step_count)
                            await queue.put(EvReasoningEnded())
                        # "buffered" events are internal to the detector

        # Finalize detector — flush remaining content
        if isinstance(detector, StreamingReasoningDetector):
            result = detector.finalize()
            if result.output_tokens:
                for t in result.output_tokens:
                    await queue.put(EvOutput(text=t))

    except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
        logger.error("Upstream connection failed: %s", exc)
        await queue.put(EvError(message=f"Upstream unavailable – {type(exc).__name__}"))

    await queue.put(EvUpstreamDone())


# ---------------------------------------------------------------------------
# Main pipeline: consumer that yields SSE bytes
# ---------------------------------------------------------------------------

async def process_stream(
    request: GatewayRequest,
    settings: Settings,
    http_client: httpx.AsyncClient,
    cached_prompt_summary: list[str] | None = None,
) -> AsyncIterator[bytes]:
    """
    Main entry point. Yields raw SSE bytes (ready to write to response).

    Pipeline:
      Phase 1: Prompt summary (from cache or parallel task)
      Phase 2: Reasoning summaries (incremental, per-step)
      Phase 3: Final output
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
    has_cached_summary = cached_prompt_summary is not None and len(cached_prompt_summary) > 0

    # ------------------------------------------------------------------
    # Start prompt summary in parallel (if not cached)
    # ------------------------------------------------------------------
    prompt_summary_task = None
    if not has_cached_summary and request.include_prompt_summary and settings.enable_prompt_summary:
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
    # Start upstream producer (pushes events to queue)
    # ------------------------------------------------------------------
    queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

    # Build upstream URL — dedicated endpoints embed the endpoint ID in the path
    if settings.upstream_endpoint_id:
        upstream_url = (
            f"{settings.upstream_base_url.rstrip('/')}/"
            f"{settings.upstream_endpoint_id}/chat/completions"
        )
        upstream_body = request.to_upstream_body(include_model=False)
        logger.debug("Using dedicated endpoint: %s", upstream_url)
    else:
        upstream_url = f"{settings.upstream_base_url.rstrip('/')}/chat/completions"
        upstream_body = request.to_upstream_body(include_model=True)

    # FriendliAI: for reasoning models, inject parse_reasoning=True (splits thinking
    # into reasoning_content delta field) and enable_thinking=True via chat_template_kwargs.
    if profile.delta_field:
        if "parse_reasoning" not in upstream_body:
            upstream_body["parse_reasoning"] = True
        if "chat_template_kwargs" not in upstream_body:
            upstream_body["chat_template_kwargs"] = {"enable_thinking": True}
        logger.debug("Injected parse_reasoning + enable_thinking for profile %s", profile.name)

    upstream_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.upstream_api_key}",
    }

    producer_task = asyncio.create_task(
        _upstream_producer(
            queue=queue,
            upstream_url=upstream_url,
            upstream_headers=upstream_headers,
            upstream_body=upstream_body,
            settings=settings,
            http_client=http_client,
            profile=profile,
        )
    )

    # ------------------------------------------------------------------
    # PHASE 1: Prompt Summary
    # ------------------------------------------------------------------
    if has_cached_summary or prompt_summary_task is not None:
        yield PhaseEvent(phase="prompt_summary", label="Prompt Summary").to_sse_bytes()
        yield _make_chunk(completion_id, model, "", role="assistant").to_sse_bytes()

        if has_cached_summary:
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

    # ------------------------------------------------------------------
    # PHASE 2 + 3: Consume queue — reasoning summaries then output
    # ------------------------------------------------------------------
    output_buffer: list[str] = []
    reasoning_step_count = 0
    reasoning_active = True  # True until we see EvReasoningEnded or EvUpstreamDone
    include_reasoning = request.include_reasoning_summary and settings.enable_reasoning_summary

    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=settings.upstream_timeout_s)
        except asyncio.TimeoutError:
            logger.error("Queue read timed out — upstream may have stalled")
            yield _make_chunk(
                completion_id, model, "[Gateway error: upstream timeout]"
            ).to_sse_bytes()
            break

        # --- Reasoning step detected ---
        if isinstance(event, EvReasoningStep):
            if include_reasoning:
                step = event.step
                # Collect tokens first — skip the step entirely if summary is empty
                step_tokens: list[str] = []
                try:
                    async for t in summariser.summarise_reasoning_step(
                        step.text, step.index, http_client
                    ):
                        step_tokens.append(t)
                except Exception as exc:
                    logger.warning("Step summary failed: %s", exc)

                if not "".join(step_tokens).strip():
                    logger.debug("Skipping empty reasoning step (index=%d)", step.index)
                else:
                    reasoning_step_count += 1
                    yield PhaseEvent(
                        phase="reasoning_summary",
                        label=f"Reasoning Step {reasoning_step_count}",
                        step=reasoning_step_count,
                    ).to_sse_bytes()
                    for t in step_tokens:
                        yield _make_chunk(completion_id, model, t).to_sse_bytes()

        # --- Reasoning block ended ---
        elif isinstance(event, EvReasoningEnded):
            reasoning_active = False

        # --- Output token ---
        elif isinstance(event, EvOutput):
            output_buffer.append(event.text)

        # --- Upstream done ---
        elif isinstance(event, EvUpstreamDone):
            break

        # --- Error ---
        elif isinstance(event, EvError):
            yield _make_chunk(
                completion_id, model, f"[Gateway error: {event.message}]"
            ).to_sse_bytes()
            # Continue reading — there might be an EvUpstreamDone following

    # Ensure producer is done
    if not producer_task.done():
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # PHASE 3: Final output
    # ------------------------------------------------------------------
    yield PhaseEvent(phase="output", label="Response").to_sse_bytes()
    for token in output_buffer:
        yield _make_chunk(completion_id, model, token).to_sse_bytes()

    # Finish
    yield _make_chunk(completion_id, model, "", finish_reason="stop").to_sse_bytes()
    yield b"data: [DONE]\n\n"
