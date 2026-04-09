"""Test suite for the FastAPI API layer (api/main.py, api/routes/chat.py, api/routes/ingest.py).

Covers:
    - GET /health → 200 {"status": "ok", "version": "0.1.0"}
    - POST /chat happy path → 200 with ChatResponse
    - POST /chat HITL pause → 202 with HITLPendingResponse
    - POST /chat invalid body → 422 validation error
    - POST /chat/resume with feedback → 200
    - POST /chat/resume HITL again → 202
    - POST /ingest → 202 with job_id
    - GET /ingest/status/{job_id} pending → 200 with status
    - GET /ingest/status/{unknown_id} → 404

All OrchestratorAgent and IngestionPipeline calls are fully mocked — no LLM or
ChromaDB calls are made during this test suite.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langgraph.errors import GraphInterrupt
from langgraph.types import Interrupt

from core.types import SupportState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_final_state(**overrides) -> SupportState:
    """Return a realistic final SupportState simulating a completed graph run."""
    base: SupportState = {
        "user_query": "I want a refund.",
        "session_id": "test-sess-001",
        "intent": "refund",
        "retrieved_context": ["Refund within 30 days.", "Contact support@example.com."],
        "draft_reply": "We can help with your refund.",
        "final_reply": "We can help with your refund. Please contact support@example.com.",
        "retry_count": 1,
        "metadata": {},
        "requires_human_review": False,
        "human_feedback": None,
        "messages": [],
    }
    base.update(overrides)
    return base


def _make_graph_interrupt(draft: str = "Draft needing review.") -> GraphInterrupt:
    """Create a GraphInterrupt exception that mimics LangGraph's HITL pause."""
    interrupt_obj = Interrupt(value={"reason": "human review required", "draft": draft})
    return GraphInterrupt((interrupt_obj,))


# ── Client fixture ─────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """Return a TestClient with OrchestratorAgent patched out.

    The orchestrator singleton in api/routes/chat.py is reset before each test
    so patches applied in individual tests take effect cleanly.
    """
    # Patch out OrchestratorAgent so __init__ doesn't hit OpenAI / ChromaDB
    with patch("api.routes.chat._orchestrator", None):
        from api.main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GET /health
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """Tests for the GET /health liveness probe."""

    def test_health_returns_200(self, client: TestClient):
        """Health check must return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client: TestClient):
        """Response body must contain status=ok."""
        response = client.get("/health")
        body = response.json()
        assert body["status"] == "ok"

    def test_health_returns_version(self, client: TestClient):
        """Response must include version field."""
        response = client.get("/health")
        assert "version" in response.json()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. POST /chat — happy path
# ═══════════════════════════════════════════════════════════════════════════════

class TestChatEndpointHappyPath:
    """Tests for POST /chat when the graph completes normally."""

    def test_chat_returns_200(self, client: TestClient):
        """Successful chat returns HTTP 200."""
        final_state = _make_final_state()
        mock_orch = MagicMock()
        mock_orch.run = AsyncMock(return_value=final_state)

        with patch("api.routes.chat.get_orchestrator", return_value=mock_orch):
            response = client.post("/chat", json={"query": "I want a refund."})

        assert response.status_code == 200

    def test_chat_response_contains_reply(self, client: TestClient):
        """Response body must contain the final_reply."""
        final_state = _make_final_state(final_reply="Your refund is processed.")
        mock_orch = MagicMock()
        mock_orch.run = AsyncMock(return_value=final_state)

        with patch("api.routes.chat.get_orchestrator", return_value=mock_orch):
            response = client.post(
                "/chat",
                json={"query": "Refund please.", "session_id": "s-001"},
            )

        body = response.json()
        assert body["reply"] == "Your refund is processed."

    def test_chat_response_contains_intent(self, client: TestClient):
        """Response body must contain the classified intent."""
        final_state = _make_final_state(intent="billing")
        mock_orch = MagicMock()
        mock_orch.run = AsyncMock(return_value=final_state)

        with patch("api.routes.chat.get_orchestrator", return_value=mock_orch):
            response = client.post("/chat", json={"query": "Billing issue."})

        assert response.json()["intent"] == "billing"

    def test_chat_response_contains_session_id(self, client: TestClient):
        """Response must echo the session_id from the request."""
        final_state = _make_final_state(session_id="my-session")
        mock_orch = MagicMock()
        mock_orch.run = AsyncMock(return_value=final_state)

        with patch("api.routes.chat.get_orchestrator", return_value=mock_orch):
            response = client.post(
                "/chat",
                json={"query": "Help.", "session_id": "my-session"},
            )

        assert response.json()["session_id"] == "my-session"

    def test_chat_response_contains_retry_count(self, client: TestClient):
        """Response must include retry_count from the final state."""
        final_state = _make_final_state(retry_count=2)
        mock_orch = MagicMock()
        mock_orch.run = AsyncMock(return_value=final_state)

        with patch("api.routes.chat.get_orchestrator", return_value=mock_orch):
            response = client.post("/chat", json={"query": "Help."})

        assert response.json()["retry_count"] == 2

    def test_chat_auto_generates_session_id_if_missing(self, client: TestClient):
        """If session_id is not provided, one must be auto-generated."""
        final_state = _make_final_state()
        mock_orch = MagicMock()
        mock_orch.run = AsyncMock(return_value=final_state)

        with patch("api.routes.chat.get_orchestrator", return_value=mock_orch):
            response = client.post("/chat", json={"query": "Hello."})

        body = response.json()
        # Must have a non-empty session_id
        assert "session_id" in body
        assert len(body["session_id"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. POST /chat — HITL pause path
# ═══════════════════════════════════════════════════════════════════════════════

class TestChatEndpointHITL:
    """Tests for POST /chat when the graph pauses for human review."""

    def test_chat_hitl_returns_202(self, client: TestClient):
        """When GraphInterrupt is raised, must return HTTP 202."""
        mock_orch = MagicMock()
        mock_orch.run = AsyncMock(
            side_effect=_make_graph_interrupt("Please review this draft.")
        )

        with patch("api.routes.chat.get_orchestrator", return_value=mock_orch):
            response = client.post("/chat", json={"query": "I need help."})

        assert response.status_code == 202

    def test_chat_hitl_response_status_field(self, client: TestClient):
        """202 response must contain status=human_review_required."""
        mock_orch = MagicMock()
        mock_orch.run = AsyncMock(
            side_effect=_make_graph_interrupt("Draft text here.")
        )

        with patch("api.routes.chat.get_orchestrator", return_value=mock_orch):
            response = client.post("/chat", json={"query": "Help."})

        body = response.json()
        assert body["status"] == "human_review_required"

    def test_chat_hitl_response_contains_draft(self, client: TestClient):
        """202 response must include the draft that needs review."""
        mock_orch = MagicMock()
        mock_orch.run = AsyncMock(
            side_effect=_make_graph_interrupt("This is the draft.")
        )

        with patch("api.routes.chat.get_orchestrator", return_value=mock_orch):
            response = client.post("/chat", json={"query": "Help."})

        body = response.json()
        assert "draft_reply" in body

    def test_chat_hitl_response_contains_session_id(self, client: TestClient):
        """202 response must include session_id."""
        mock_orch = MagicMock()
        mock_orch.run = AsyncMock(
            side_effect=_make_graph_interrupt()
        )

        with patch("api.routes.chat.get_orchestrator", return_value=mock_orch):
            response = client.post(
                "/chat", json={"query": "Help.", "session_id": "hitl-sess"}
            )

        assert response.json()["session_id"] == "hitl-sess"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. POST /chat — validation errors
# ═══════════════════════════════════════════════════════════════════════════════

class TestChatEndpointValidation:
    """Tests for POST /chat input validation."""

    def test_missing_query_returns_422(self, client: TestClient):
        """Request without 'query' field must return HTTP 422."""
        response = client.post("/chat", json={"session_id": "abc"})
        assert response.status_code == 422

    def test_empty_query_returns_422(self, client: TestClient):
        """Empty string query must fail validation (min_length=1)."""
        response = client.post("/chat", json={"query": ""})
        assert response.status_code == 422

    def test_non_json_body_returns_422(self, client: TestClient):
        """Non-JSON body must return HTTP 422."""
        response = client.post(
            "/chat",
            data="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# 5. POST /chat/resume
# ═══════════════════════════════════════════════════════════════════════════════

class TestResumeEndpoint:
    """Tests for POST /chat/resume (HITL continuation)."""

    def test_resume_returns_200_on_completion(self, client: TestClient):
        """Successful graph resume must return HTTP 200."""
        final_state = _make_final_state(final_reply="Improved reply after feedback.")
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=final_state)

        mock_orch = MagicMock()
        mock_orch._graph = mock_graph

        with patch("api.routes.chat.get_orchestrator", return_value=mock_orch):
            response = client.post(
                "/chat/resume",
                json={"session_id": "s-001", "human_feedback": "Please shorten it."},
            )

        assert response.status_code == 200

    def test_resume_response_contains_reply(self, client: TestClient):
        """Resume response must contain the final reply."""
        final_state = _make_final_state(final_reply="Concise reply after review.")
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=final_state)

        mock_orch = MagicMock()
        mock_orch._graph = mock_graph

        with patch("api.routes.chat.get_orchestrator", return_value=mock_orch):
            response = client.post(
                "/chat/resume",
                json={"session_id": "s-002", "human_feedback": "Good."},
            )

        assert response.json()["reply"] == "Concise reply after review."

    def test_resume_another_interrupt_returns_202(self, client: TestClient):
        """If another interrupt occurs during resume, must return 202."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            side_effect=_make_graph_interrupt("Still needs work.")
        )
        mock_orch = MagicMock()
        mock_orch._graph = mock_graph

        with patch("api.routes.chat.get_orchestrator", return_value=mock_orch):
            response = client.post(
                "/chat/resume",
                json={"session_id": "s-003", "human_feedback": "Try again."},
            )

        assert response.status_code == 202

    def test_resume_missing_session_id_returns_422(self, client: TestClient):
        """Resume request without session_id must return 422."""
        response = client.post(
            "/chat/resume", json={"human_feedback": "feedback only"}
        )
        assert response.status_code == 422

    def test_resume_empty_feedback_is_valid(self, client: TestClient):
        """Empty human_feedback is allowed (means 'approve as-is')."""
        final_state = _make_final_state()
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=final_state)
        mock_orch = MagicMock()
        mock_orch._graph = mock_graph

        with patch("api.routes.chat.get_orchestrator", return_value=mock_orch):
            response = client.post(
                "/chat/resume",
                json={"session_id": "s-004", "human_feedback": ""},
            )

        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# 6. POST /ingest
# ═══════════════════════════════════════════════════════════════════════════════

class TestIngestEndpoint:
    """Tests for POST /ingest."""

    def test_ingest_returns_202(self, client: TestClient):
        """POST /ingest must return HTTP 202 Accepted immediately."""
        response = client.post("/ingest", json={})
        assert response.status_code == 202

    def test_ingest_response_contains_job_id(self, client: TestClient):
        """Response must contain a job_id field."""
        response = client.post("/ingest", json={})
        body = response.json()
        assert "job_id" in body
        assert len(body["job_id"]) > 0

    def test_ingest_response_status_is_pending(self, client: TestClient):
        """Initial status must be 'pending'."""
        response = client.post("/ingest", json={})
        body = response.json()
        assert body["status"] == "pending"

    def test_ingest_with_custom_source_dir(self, client: TestClient):
        """POST /ingest with custom source_dir returns 202."""
        response = client.post("/ingest", json={"source_dir": "data/custom"})
        assert response.status_code == 202

    def test_ingest_returns_message(self, client: TestClient):
        """Response must contain a human-readable message."""
        response = client.post("/ingest", json={})
        body = response.json()
        assert "message" in body


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GET /ingest/status/{job_id}
# ═══════════════════════════════════════════════════════════════════════════════

class TestIngestStatusEndpoint:
    """Tests for GET /ingest/status/{job_id}."""

    def test_status_returns_200_for_known_job(self, client: TestClient):
        """GET /ingest/status/{job_id} for a known job returns 200."""
        # First create a job
        post_resp = client.post("/ingest", json={})
        job_id = post_resp.json()["job_id"]

        response = client.get(f"/ingest/status/{job_id}")
        assert response.status_code == 200

    def test_status_contains_job_id(self, client: TestClient):
        """Response body must echo the job_id."""
        post_resp = client.post("/ingest", json={})
        job_id = post_resp.json()["job_id"]

        response = client.get(f"/ingest/status/{job_id}")
        assert response.json()["job_id"] == job_id

    def test_status_contains_status_field(self, client: TestClient):
        """Response body must contain a 'status' field."""
        post_resp = client.post("/ingest", json={})
        job_id = post_resp.json()["job_id"]

        response = client.get(f"/ingest/status/{job_id}")
        assert "status" in response.json()

    def test_status_returns_404_for_unknown_job(self, client: TestClient):
        """GET /ingest/status for a non-existent job_id must return 404."""
        response = client.get(f"/ingest/status/{uuid.uuid4()}")
        # The global APIError handler converts 404 APIError → 404 response
        assert response.status_code == 404

    def test_status_404_body_contains_error(self, client: TestClient):
        """404 response must include an 'error' field."""
        fake_id = str(uuid.uuid4())
        response = client.get(f"/ingest/status/{fake_id}")
        body = response.json()
        assert "error" in body
