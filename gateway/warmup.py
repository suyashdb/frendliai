"""
Session warmup manager.

Pre-warms sessions by:
  1. Pre-computing the prompt summary so it's instant when /chat/completions fires
  2. Sending the system prompt to the upstream to prime KV cache (prefix caching)

Sessions are stored in-memory with TTL-based expiry.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

import httpx

from gateway.config import Settings
from gateway.summarizer import Summariser

logger = logging.getLogger(__name__)


@dataclass
class WarmSession:
    """A pre-warmed session ready for fast /chat/completions."""

    session_id: str
    model: str
    system_prompt: str | None
    created_at: float = field(default_factory=time.time)

    # Pre-computed results
    prompt_summary_tokens: list[str] = field(default_factory=list)
    prompt_summary_ready: bool = False
    kv_cache_warmed: bool = False

    # Internal
    _summary_task: asyncio.Task | None = field(default=None, repr=False)
    _kv_task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def age_s(self) -> float:
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        return self.age_s > 300  # 5 min TTL


class SessionManager:
    """
    In-memory session store with background warmup tasks.

    Lifecycle:
      1. Client calls POST /v1/sessions/warm with model + system_prompt
      2. Manager creates a session, kicks off:
         a. Prompt summary pre-computation (for the system prompt context)
         b. KV cache warming request to upstream
      3. Client calls POST /v1/chat/completions with session_id
      4. Gateway uses cached summary → near-zero TTFT for Phase 1
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: dict[str, WarmSession] = {}
        self._summariser = Summariser(settings)
        self._cleanup_interval = 60  # seconds

    async def create_session(
        self,
        model: str,
        system_prompt: str | None,
        messages: list[dict] | None,
        http_client: httpx.AsyncClient,
    ) -> WarmSession:
        """
        Create and warm a new session.

        Kicks off background tasks for:
          - Prompt summary pre-computation (if system_prompt or messages provided)
          - KV cache warming (sends system prompt to upstream)
        """
        session_id = f"sess-{uuid.uuid4().hex[:16]}"
        session = WarmSession(
            session_id=session_id,
            model=model,
            system_prompt=system_prompt,
        )

        # --- Background task 1: Pre-compute prompt summary ---
        warmup_messages = []
        if system_prompt:
            warmup_messages.append({"role": "system", "content": system_prompt})
        if messages:
            warmup_messages.extend(messages)

        if warmup_messages:
            session._summary_task = asyncio.create_task(
                self._precompute_summary(session, warmup_messages, http_client)
            )

        # --- Background task 2: KV cache warming ---
        if system_prompt:
            session._kv_task = asyncio.create_task(
                self._warm_kv_cache(session, http_client)
            )

        self._sessions[session_id] = session
        logger.info(
            "Session %s created for model=%s (system_prompt=%s)",
            session_id,
            model,
            bool(system_prompt),
        )
        return session

    def get_session(self, session_id: str) -> WarmSession | None:
        """Retrieve a session, returning None if expired or not found."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired:
            logger.info("Session %s expired, removing", session_id)
            self._sessions.pop(session_id, None)
            return None
        return session

    def remove_session(self, session_id: str) -> bool:
        """Explicitly remove a session."""
        return self._sessions.pop(session_id, None) is not None

    async def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count removed."""
        expired = [
            sid for sid, s in self._sessions.items() if s.is_expired
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))
        return len(expired)

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    # ------------------------------------------------------------------
    # Background warmup tasks
    # ------------------------------------------------------------------

    async def _precompute_summary(
        self,
        session: WarmSession,
        messages: list[dict],
        http_client: httpx.AsyncClient,
    ) -> None:
        """Pre-compute the prompt summary and cache it in the session."""
        try:
            tokens: list[str] = []
            async for t in self._summariser.summarise_prompt(messages, http_client):
                tokens.append(t)
            session.prompt_summary_tokens = tokens
            session.prompt_summary_ready = True
            logger.info(
                "Session %s: prompt summary pre-computed (%d tokens)",
                session.session_id,
                len(tokens),
            )
        except Exception as exc:
            logger.warning(
                "Session %s: prompt summary pre-computation failed: %s",
                session.session_id,
                exc,
            )
            session.prompt_summary_ready = False

    async def _warm_kv_cache(
        self,
        session: WarmSession,
        http_client: httpx.AsyncClient,
    ) -> None:
        """
        Send a minimal request with the system prompt to the upstream to prime
        the KV cache.  Many inference engines (vLLM, TensorRT-LLM, FriendliAI
        Engine) support prefix caching — the system prompt tokens won't need
        to be recomputed when the real request arrives.

        We send a tiny max_tokens=1 request and discard the output.
        """
        if not session.system_prompt:
            return

        url = f"{self._settings.upstream_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._settings.upstream_api_key}",
        }
        body = {
            "model": session.model,
            "messages": [
                {"role": "system", "content": session.system_prompt},
                {"role": "user", "content": "hi"},  # minimal user msg
            ],
            "max_tokens": 1,
            "stream": False,
        }

        try:
            resp = await http_client.post(
                url,
                json=body,
                headers=headers,
                timeout=self._settings.upstream_connect_timeout_s + 5.0,
            )
            if resp.status_code == 200:
                session.kv_cache_warmed = True
                logger.info("Session %s: KV cache warmed", session.session_id)
            else:
                logger.warning(
                    "Session %s: KV warm returned %d",
                    session.session_id,
                    resp.status_code,
                )
        except Exception as exc:
            logger.warning(
                "Session %s: KV cache warming failed: %s",
                session.session_id,
                exc,
            )
