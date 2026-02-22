"""
Mock upstream /chat/completions server.

Simulates a reasoning-capable LLM that:
  - Emits <think>...</think> reasoning tokens
  - Followed by the actual output tokens
  - Supports streaming SSE

Also doubles as a mock summariser endpoint (returns short summaries).

Run:
    python mock_upstream.py
    # Starts on port 9000
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(title="Mock Upstream LLM")

# ---------------------------------------------------------------------------
# Simulated responses for different scenarios
# ---------------------------------------------------------------------------

REASONING_RESPONSE = {
    "reasoning": (
        "<think>\n"
        "The user is asking about the capital of France.\n"
        "Let me recall: France is a country in Western Europe.\n"
        "Its capital city is Paris, which has been the capital since the 10th century.\n"
        "Paris is also the largest city in France with a population of about 2.1 million "
        "in the city proper.\n"
        "I'm confident the answer is Paris.\n"
        "</think>"
    ),
    "output": (
        "The capital of France is **Paris**. It is located in the north-central "
        "part of the country along the Seine River. Paris has served as the French "
        "capital since the late 10th century and is today one of the world's major "
        "centres of finance, diplomacy, commerce, culture, fashion, and gastronomy."
    ),
}

SIMPLE_RESPONSE = (
    "This is a simple response without any reasoning tokens. "
    "The gateway should pass this through directly."
)

SUMMARY_RESPONSE_PROMPT = "The user asks about the capital of France."
SUMMARY_RESPONSE_REASONING = (
    "The model recalled geographic knowledge about France, confirmed Paris "
    "as the capital, and noted its historical significance."
)


def _token_stream(text: str, model: str, chunk_size: int = 4):
    """Split text into small chunks to simulate token-level streaming."""
    completion_id = f"chatcmpl-mock-{uuid.uuid4().hex[:8]}"
    tokens = []
    for i in range(0, len(text), chunk_size):
        tokens.append(text[i : i + chunk_size])
    return completion_id, tokens


async def _stream_response(text: str, model: str, delay: float = 0.02):
    """Generate SSE chunks for a text response."""
    completion_id, tokens = _token_stream(text, model)

    # Initial role chunk
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    await asyncio.sleep(delay)

    # Content chunks
    for token in tokens:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(delay)

    # Final chunk
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "mock-model")
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Detect if this is a summarisation request (from the gateway's summariser)
    is_summary_request = False
    for msg in messages:
        content = msg.get("content", "")
        if "concise summariser" in content.lower():
            is_summary_request = True
            break

    # Choose response
    if is_summary_request:
        # Check if it's a reasoning summary or prompt summary
        user_msg = messages[-1].get("content", "") if messages else ""
        if "<think>" in user_msg or "chain-of-thought" in user_msg.lower():
            text = SUMMARY_RESPONSE_REASONING
        else:
            text = SUMMARY_RESPONSE_PROMPT
    elif "deepseek" in model.lower() or "qwq" in model.lower() or "think" in model.lower():
        text = REASONING_RESPONSE["reasoning"] + REASONING_RESPONSE["output"]
    else:
        # Check if user asked something that triggers reasoning
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        if last_user_msg:
            text = REASONING_RESPONSE["reasoning"] + REASONING_RESPONSE["output"]
        else:
            text = SIMPLE_RESPONSE

    if not stream:
        return JSONResponse({
            "id": f"chatcmpl-mock-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": len(text) // 4, "total_tokens": 10 + len(text) // 4},
        })

    return StreamingResponse(
        _stream_response(text, model, delay=0.015),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "deepseek-r1", "object": "model", "owned_by": "mock"},
            {"id": "qwen-qwq-32b", "object": "model", "owned_by": "mock"},
            {"id": "gpt-4o-mini", "object": "model", "owned_by": "mock"},
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
