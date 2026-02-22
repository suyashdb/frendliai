"""
Gateway server – main entry point.

A FastAPI application that proxies /chat/completions requests through an
intelligent reasoning-aware pipeline:

  Request → [Prompt Summary ∥ Upstream Stream] → Reasoning Detection
          → Reasoning Summary → Final Output → Client (SSE)

Run:
    python -m gateway.server
    # or
    uvicorn gateway.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from gateway.config import Settings
from gateway.models import GatewayRequest, WarmupRequest, WarmupResponse
from gateway.stream_processor import process_stream
from gateway.warmup import SessionManager

logger = logging.getLogger("gateway")

# ---------------------------------------------------------------------------
# Application lifespan – manages shared httpx client + session manager
# ---------------------------------------------------------------------------

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create shared httpx.AsyncClient and session manager."""
    app.state.http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        ),
        follow_redirects=True,
    )
    app.state.session_manager = SessionManager(settings)

    # Background session cleanup task
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(60)
            try:
                await app.state.session_manager.cleanup_expired()
            except Exception:
                pass

    cleanup_task = asyncio.create_task(_cleanup_loop())

    logger.info("Gateway started – upstream: %s", settings.upstream_base_url)
    yield
    cleanup_task.cancel()
    await app.state.http_client.aclose()
    logger.info("Gateway shut down")


app = FastAPI(
    title="FriendliAI Reasoning Gateway",
    version="1.0.0",
    description=(
        "Streaming gateway that enriches /chat/completions responses "
        "with prompt and reasoning summaries."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible /chat/completions endpoint with reasoning enrichment.

    Accepts the standard request body plus optional gateway extensions:
      - include_prompt_summary (bool): include Phase 1
      - include_reasoning_summary (bool): include Phase 2
      - reasoning_profile (str): override auto-detected model profile
      - session_id (str): attach to a pre-warmed session for faster TTFT
    """
    body = await request.json()
    try:
        gw_request = GatewayRequest(**body)
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(exc), "type": "invalid_request_error"}},
        )

    client: httpx.AsyncClient = request.app.state.http_client
    session_mgr: SessionManager = request.app.state.session_manager

    # --- Check for pre-warmed session ---
    cached_prompt_summary = None
    if gw_request.session_id:
        session = session_mgr.get_session(gw_request.session_id)
        if session and session.prompt_summary_ready:
            cached_prompt_summary = session.prompt_summary_tokens
            logger.info(
                "Using warm session %s (summary cached, kv_warm=%s)",
                session.session_id,
                session.kv_cache_warmed,
            )
        elif session:
            logger.info(
                "Session %s found but summary not ready yet – computing live",
                session.session_id,
            )
        else:
            logger.warning("Session %s not found or expired", gw_request.session_id)

    return StreamingResponse(
        process_stream(gw_request, settings, client, cached_prompt_summary),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/sessions/warm", response_model=WarmupResponse)
async def warm_session(request: Request):
    """
    Pre-warm a session for faster subsequent /chat/completions calls.

    This endpoint:
      1. Pre-computes the prompt summary (cached for instant Phase 1 delivery)
      2. Sends a minimal request to upstream to prime the KV cache

    Returns a session_id to pass in the /chat/completions request body.
    Sessions expire after 5 minutes.
    """
    body = await request.json()
    try:
        warmup_req = WarmupRequest(**body)
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(exc), "type": "invalid_request_error"}},
        )

    client: httpx.AsyncClient = request.app.state.http_client
    session_mgr: SessionManager = request.app.state.session_manager

    messages = None
    if warmup_req.messages:
        messages = [m.model_dump(exclude_none=True) for m in warmup_req.messages]

    session = await session_mgr.create_session(
        model=warmup_req.model,
        system_prompt=warmup_req.system_prompt,
        messages=messages,
        http_client=client,
    )

    return WarmupResponse(
        session_id=session.session_id,
        model=session.model,
        status="warming",
        prompt_summary_ready=session.prompt_summary_ready,
        kv_cache_warmed=session.kv_cache_warmed,
    )


@app.get("/v1/sessions/{session_id}")
async def get_session_status(session_id: str, request: Request):
    """Check the warmup status of a session."""
    session_mgr: SessionManager = request.app.state.session_manager
    session = session_mgr.get_session(session_id)

    if session is None:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "Session not found or expired"}},
        )

    return {
        "session_id": session.session_id,
        "model": session.model,
        "status": "ready" if session.prompt_summary_ready else "warming",
        "prompt_summary_ready": session.prompt_summary_ready,
        "kv_cache_warmed": session.kv_cache_warmed,
        "age_seconds": round(session.age_s, 1),
        "ttl_seconds": max(0, 300 - int(session.age_s)),
    }


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Explicitly delete a session."""
    session_mgr: SessionManager = request.app.state.session_manager
    removed = session_mgr.remove_session(session_id)
    if not removed:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "Session not found"}},
        )
    return {"status": "deleted", "session_id": session_id}


@app.get("/health")
async def health(request: Request):
    session_mgr: SessionManager = request.app.state.session_manager
    return {
        "status": "ok",
        "upstream": settings.upstream_base_url,
        "active_sessions": session_mgr.active_count,
    }


@app.get("/v1/models")
async def list_models(request: Request):
    """Proxy model listing from upstream."""
    client: httpx.AsyncClient = request.app.state.http_client
    try:
        resp = await client.get(
            f"{settings.upstream_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {settings.upstream_api_key}"},
            timeout=10.0,
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Upstream unreachable: {exc}"}},
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    uvicorn.run(
        "gateway.server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
