"""Integration tests for the gateway API."""

import pytest
from fastapi.testclient import TestClient

from gateway.server import app


class TestHealthEndpoint:
    def test_health(self):
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert "active_sessions" in data


class TestRequestValidation:
    def test_missing_model(self):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 400

    def test_missing_messages(self):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "test-model"},
            )
            assert resp.status_code == 400

    def test_valid_minimal_request(self):
        """Valid request should return 200 streaming response (may error on upstream)."""
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]


class TestSessionWarmup:
    def test_create_warm_session(self):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/sessions/warm",
                json={"model": "deepseek-r1", "system_prompt": "You are a helpful assistant."},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "session_id" in data
            assert data["model"] == "deepseek-r1"
            assert data["status"] == "warming"
            assert data["ttl_seconds"] == 300

    def test_create_warm_session_no_system(self):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/sessions/warm",
                json={"model": "deepseek-r1"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "session_id" in data

    def test_get_session_status(self):
        with TestClient(app) as client:
            create_resp = client.post(
                "/v1/sessions/warm",
                json={"model": "deepseek-r1"},
            )
            session_id = create_resp.json()["session_id"]

            status_resp = client.get(f"/v1/sessions/{session_id}")
            assert status_resp.status_code == 200
            data = status_resp.json()
            assert data["session_id"] == session_id
            assert "prompt_summary_ready" in data
            assert "kv_cache_warmed" in data
            assert data["ttl_seconds"] <= 300

    def test_get_nonexistent_session(self):
        with TestClient(app) as client:
            resp = client.get("/v1/sessions/nonexistent-session-id")
            assert resp.status_code == 404

    def test_delete_session(self):
        with TestClient(app) as client:
            create_resp = client.post(
                "/v1/sessions/warm",
                json={"model": "deepseek-r1"},
            )
            session_id = create_resp.json()["session_id"]

            del_resp = client.delete(f"/v1/sessions/{session_id}")
            assert del_resp.status_code == 200

            get_resp = client.get(f"/v1/sessions/{session_id}")
            assert get_resp.status_code == 404

    def test_delete_nonexistent_session(self):
        with TestClient(app) as client:
            resp = client.delete("/v1/sessions/nonexistent-id")
            assert resp.status_code == 404

    def test_completions_with_session_id(self):
        """Completions with a session_id should still return 200 streaming."""
        with TestClient(app) as client:
            create_resp = client.post(
                "/v1/sessions/warm",
                json={"model": "test-model"},
            )
            session_id = create_resp.json()["session_id"]

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "session_id": session_id,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

    def test_completions_with_invalid_session_id(self):
        """Invalid session_id should gracefully fall through to live computation."""
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "session_id": "invalid-session-id",
                },
            )
            assert resp.status_code == 200

    def test_warmup_missing_model(self):
        with TestClient(app) as client:
            resp = client.post(
                "/v1/sessions/warm",
                json={},
            )
            assert resp.status_code == 400
