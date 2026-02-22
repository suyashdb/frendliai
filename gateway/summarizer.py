"""
Streaming summariser.

Calls a (possibly different) LLM to produce concise summaries of:
  1. The user's prompt / conversation
  2. The model's internal reasoning

Both summaries are streamed token-by-token so the gateway can forward
them to the client with minimal latency.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from gateway.config import Settings

logger = logging.getLogger(__name__)

PROMPT_SUMMARY_SYSTEM = (
    "You are a concise summariser. Summarise the user's request in 1-2 sentences. "
    "Focus on what the user wants done. Do NOT answer the question – just summarise it. "
    "Be direct and brief."
)

REASONING_SUMMARY_SYSTEM = (
    "You are a concise summariser. You will receive the internal chain-of-thought "
    "reasoning produced by an AI model. Summarise the key reasoning steps and "
    "conclusions in 2-4 sentences. Omit false starts and redundant explorations. "
    "Be direct and brief."
)


class Summariser:
    """Async streaming summariser backed by a /chat/completions endpoint."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.resolved_summariser_base_url.rstrip("/")
        self._api_key = settings.resolved_summariser_api_key
        self._model = settings.summariser_model
        self._timeout = settings.summariser_timeout_s
        self._max_retries = settings.max_retries

    async def summarise_prompt(
        self,
        messages: list[dict],
        client: httpx.AsyncClient,
    ) -> AsyncIterator[str]:
        """Stream a summary of the user prompt."""
        # Build a flat text representation of the conversation
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            parts.append(f"{role}: {content}")
        conversation_text = "\n".join(parts)

        async for token in self._stream_summary(
            system=PROMPT_SUMMARY_SYSTEM,
            user_text=conversation_text,
            client=client,
        ):
            yield token

    async def summarise_reasoning(
        self,
        reasoning_text: str,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[str]:
        """Stream a summary of the model's reasoning."""
        if not reasoning_text.strip():
            return
        # Truncate very long reasoning to avoid blowing context
        truncated = reasoning_text[:12_000]
        async for token in self._stream_summary(
            system=REASONING_SUMMARY_SYSTEM,
            user_text=truncated,
            client=client,
        ):
            yield token

    async def _stream_summary(
        self,
        system: str,
        user_text: str,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[str]:
        """Low-level: call the summariser model with streaming."""
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            "stream": True,
            "max_tokens": 256,
            "temperature": 0.3,
        }

        attempt = 0
        while attempt <= self._max_retries:
            try:
                async with client.stream(
                    "POST",
                    url,
                    json=body,
                    headers=headers,
                    timeout=self._timeout,
                ) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        logger.warning(
                            "Summariser returned %d: %s",
                            resp.status_code,
                            error_body[:500],
                        )
                        attempt += 1
                        continue

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            return
                        try:
                            chunk = json.loads(payload)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            token = delta.get("content")
                            if token:
                                yield token
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue
                    return  # stream completed normally

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                logger.warning(
                    "Summariser attempt %d failed: %s", attempt + 1, exc
                )
                attempt += 1

        logger.error("Summariser exhausted all retries – skipping summary")
