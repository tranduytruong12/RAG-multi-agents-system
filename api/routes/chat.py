# FILE: customer_support_rag/api/routes/chat.py
"""Chat router: POST /chat, POST /chat/resume, GET /health.

POST /chat
    Accepts a user query and session_id. Builds a SupportState, invokes the
    OrchestratorAgent graph, and returns the final reply.

    If the graph pauses for HITL review (LangGraph raises GraphInterrupt),
    the handler returns HTTP 202 Accepted with a state snapshot so the caller
    knows to submit feedback via POST /chat/resume.

POST /chat/resume
    Accepts session_id and human_feedback. Resumes a paused LangGraph graph
    from its MemorySaver checkpoint by issuing Command(resume=human_feedback).

GET /health
    Lightweight liveness probe used by Docker health checks and load balancers.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from pydantic import BaseModel, Field

from agents.orchestrator import OrchestratorAgent
from core.exceptions import APIError
from core.logging import bind_request_context, get_logger
from core.types import SupportState

logger: structlog.BoundLogger = get_logger(__name__)

router = APIRouter(tags=["chat"])

# Module-level singleton — built once, reused across requests.
# In tests this is replaced via dependency override or direct monkeypatching.
_orchestrator: OrchestratorAgent | None = None


def get_orchestrator() -> OrchestratorAgent:
    """Return the shared OrchestratorAgent singleton, creating it on first call."""
    global _orchestrator  # noqa: PLW0603
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
    return _orchestrator


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Input schema for POST /chat."""

    query: str = Field(..., min_length=1, description="User's natural-language query.")
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Conversation session identifier. Auto-generated if not provided.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional caller-supplied metadata (trace_id, user_id, etc.).",
    )


class ChatResponse(BaseModel):
    """Output schema for a successfully completed POST /chat."""

    reply: str = Field(..., description="The QA-approved final reply to the user.")
    intent: str = Field(..., description="Classified intent of the query.")
    session_id: str = Field(..., description="Echo of the session_id for client tracking.")
    retry_count: int = Field(..., description="Number of draft-QA cycles that ran.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class HITLPendingResponse(BaseModel):
    """Output schema for a 202 Accepted HITL pause."""

    status: str = Field(default="human_review_required")
    session_id: str
    draft_reply: str = Field(..., description="The draft that needs human review.")
    message: str = Field(
        default="Submit your feedback via POST /chat/resume to continue."
    )


class ResumeRequest(BaseModel):
    """Input schema for POST /chat/resume."""

    session_id: str = Field(..., description="The session_id from the original /chat call.")
    human_feedback: str = Field(
        default="",
        description="Human reviewer's feedback. Empty string = approve the draft as-is.",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    """Return API health status.

    Used by Docker HEALTHCHECK, Kubernetes liveness probes, and load balancers.
    Does NOT check downstream services (ChromaDB, OpenAI) — this is intentional
    to avoid cascading failures on startup.
    """
    return {"status": "ok", "version": "0.1.0"}


@router.post(
    "/chat",
    summary="Submit a customer support query",
    responses={
        200: {"description": "Final reply produced successfully."},
        202: {"description": "Graph paused — human review required."},
        422: {"description": "Validation error in request body."},
        500: {"description": "Internal server error."},
    },
)
async def chat(request: ChatRequest) -> JSONResponse:
    """Handle a customer support query end-to-end.

    Builds a SupportState, runs the LangGraph orchestrator, and returns:
    - HTTP 200 + ChatResponse if the graph completes normally.
    - HTTP 202 + HITLPendingResponse if the graph pauses for human review.

    A unique `request_id` is bound to the structlog context so every log line
    produced during this request automatically includes it.

    Args:
        request: Validated ChatRequest with query, session_id, optional metadata.

    Returns:
        JSONResponse with ChatResponse (200) or HITLPendingResponse (202).

    Raises:
        APIError: Translated to JSON error response by the global handler.
    """
    request_id = str(uuid.uuid4())
    bind_request_context(request_id=request_id, session_id=request.session_id)

    logger.info(
        "chat.request.received",
        session_id=request.session_id,
        query_preview=request.query[:80],
    )

    initial_state: SupportState = {
        "user_query": request.query,
        "session_id": request.session_id,
        "intent": "",
        "retrieved_context": [],
        "draft_reply": "",
        "final_reply": "",
        "retry_count": 0,
        "metadata": request.metadata,
        "requires_human_review": False,
        "human_feedback": None,
        "messages": [],
    }

    try:
        orchestrator = get_orchestrator()
        final_state: SupportState = await orchestrator.run(initial_state)

    except GraphInterrupt as exc:
        # LangGraph paused the graph — extract the interrupt payload.
        # exc.args[0] is a tuple of Interrupt objects; each has a .value attribute.
        interrupts = exc.args[0] if exc.args else []
        draft = ""
        if interrupts:
            payload = interrupts[0].value if hasattr(interrupts[0], "value") else {}
            draft = payload.get("draft", "") if isinstance(payload, dict) else ""

        logger.info(
            "chat.hitl.paused",
            session_id=request.session_id,
            request_id=request_id,
        )
        response_body = HITLPendingResponse(
            session_id=request.session_id,
            draft_reply=draft,
        )
        return JSONResponse(status_code=202, content=response_body.model_dump())

    except Exception as exc:
        logger.error("chat.error", error=str(exc), session_id=request.session_id)
        raise APIError(f"Chat processing failed: {exc}", status_code=500) from exc

    logger.info(
        "chat.request.complete",
        session_id=request.session_id,
        intent=final_state.get("intent"),
        retry_count=final_state.get("retry_count"),
    )

    response = ChatResponse(
        reply=final_state.get("final_reply", ""),
        intent=final_state.get("intent", ""),
        session_id=request.session_id,
        retry_count=final_state.get("retry_count", 0),
        metadata=final_state.get("metadata", {}),
    )
    return JSONResponse(status_code=200, content=response.model_dump())


@router.post(
    "/chat/resume",
    summary="Resume a paused HITL graph with human feedback",
    responses={
        200: {"description": "Graph resumed and completed."},
        202: {"description": "Graph still paused after resume (another interrupt)."},
        422: {"description": "Validation error."},
        500: {"description": "Internal server error."},
    },
)
async def resume_chat(request: ResumeRequest) -> JSONResponse:
    """Resume a LangGraph graph that was paused by a HITL interrupt.

    LangGraph stores the checkpoint under the session_id thread_id in MemorySaver.
    We resume by calling ainvoke with Command(resume=human_feedback) and the same
    config dict used in the original run() call.

    Args:
        request: ResumeRequest with session_id and human_feedback string.

    Returns:
        JSONResponse with ChatResponse (200) if the graph completes, or
        HITLPendingResponse (202) if another interrupt occurs.
    """
    request_id = str(uuid.uuid4())
    bind_request_context(request_id=request_id, session_id=request.session_id)

    logger.info(
        "chat.resume.received",
        session_id=request.session_id,
        has_feedback=bool(request.human_feedback),
    )

    config = {"configurable": {"thread_id": request.session_id}}

    try:
        orchestrator = get_orchestrator()
        # Resume the paused graph by injecting the Command with human feedback.
        final_state: SupportState = await orchestrator._graph.ainvoke(
            Command(resume=request.human_feedback),
            config=config,
        )

    except GraphInterrupt as exc:
        interrupts = exc.args[0] if exc.args else []
        draft = ""
        if interrupts:
            payload = interrupts[0].value if hasattr(interrupts[0], "value") else {}
            draft = payload.get("draft", "") if isinstance(payload, dict) else ""

        logger.info("chat.resume.hitl.paused", session_id=request.session_id)
        response_body = HITLPendingResponse(
            session_id=request.session_id,
            draft_reply=draft,
        )
        return JSONResponse(status_code=202, content=response_body.model_dump())

    except Exception as exc:
        logger.error("chat.resume.error", error=str(exc))
        raise APIError(f"Resume failed: {exc}", status_code=500) from exc

    logger.info(
        "chat.resume.complete",
        session_id=request.session_id,
        intent=final_state.get("intent"),
    )

    response = ChatResponse(
        reply=final_state.get("final_reply", ""),
        intent=final_state.get("intent", ""),
        session_id=request.session_id,
        retry_count=final_state.get("retry_count", 0),
        metadata=final_state.get("metadata", {}),
    )
    return JSONResponse(status_code=200, content=response.model_dump())
