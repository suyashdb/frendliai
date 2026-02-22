"""Tests for session warmup manager."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from gateway.config import Settings
from gateway.warmup import SessionManager, WarmSession


class TestWarmSession:
    """Test WarmSession dataclass."""

    def test_session_creation(self):
        session = WarmSession(session_id="test-123", model="deepseek-r1", system_prompt=None)
        assert session.session_id == "test-123"
        assert session.model == "deepseek-r1"
        assert not session.prompt_summary_ready
        assert not session.kv_cache_warmed
        assert not session.is_expired

    def test_session_expiry(self):
        import time
        session = WarmSession(
            session_id="test-expired",
            model="deepseek-r1",
            system_prompt=None,
            created_at=time.time() - 400,  # 400s ago, TTL is 300s
        )
        assert session.is_expired

    def test_session_not_expired(self):
        session = WarmSession(session_id="test-fresh", model="deepseek-r1", system_prompt=None)
        assert not session.is_expired
        assert session.age_s < 5  # just created


class TestSessionManager:
    """Test SessionManager in-memory store."""

    def _make_manager(self) -> SessionManager:
        settings = Settings()
        return SessionManager(settings)

    @pytest.mark.asyncio
    async def test_create_session_no_warmup(self):
        """Create a session without system prompt — no background tasks."""
        mgr = self._make_manager()
        mock_client = AsyncMock()

        session = await mgr.create_session(
            model="test-model",
            system_prompt=None,
            messages=None,
            http_client=mock_client,
        )

        assert session.session_id.startswith("sess-")
        assert session.model == "test-model"
        assert mgr.active_count == 1

    @pytest.mark.asyncio
    async def test_get_session(self):
        mgr = self._make_manager()
        mock_client = AsyncMock()

        session = await mgr.create_session(
            model="test-model",
            system_prompt=None,
            messages=None,
            http_client=mock_client,
        )

        retrieved = mgr.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_get_nonexistent_session(self):
        mgr = self._make_manager()
        assert mgr.get_session("nonexistent-id") is None

    @pytest.mark.asyncio
    async def test_get_expired_session_returns_none(self):
        mgr = self._make_manager()
        mock_client = AsyncMock()

        session = await mgr.create_session(
            model="test-model",
            system_prompt=None,
            messages=None,
            http_client=mock_client,
        )
        # Force expiry
        import time
        session.created_at = time.time() - 400

        assert mgr.get_session(session.session_id) is None
        assert mgr.active_count == 0  # auto-removed

    @pytest.mark.asyncio
    async def test_remove_session(self):
        mgr = self._make_manager()
        mock_client = AsyncMock()

        session = await mgr.create_session(
            model="test-model",
            system_prompt=None,
            messages=None,
            http_client=mock_client,
        )

        assert mgr.remove_session(session.session_id) is True
        assert mgr.active_count == 0
        assert mgr.remove_session(session.session_id) is False  # already gone

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        mgr = self._make_manager()
        mock_client = AsyncMock()

        s1 = await mgr.create_session("m1", None, None, mock_client)
        s2 = await mgr.create_session("m2", None, None, mock_client)

        # Expire s1
        import time
        s1.created_at = time.time() - 400

        removed = await mgr.cleanup_expired()
        assert removed == 1
        assert mgr.active_count == 1
        assert mgr.get_session(s2.session_id) is not None
