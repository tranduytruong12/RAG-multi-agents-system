# FILE: customer_support_rag/core/types.py
"""CRITICAL SHARED CONTRACTS — Fully defined, zero stubs.

These types are the single source of truth used across ALL phases and agents.
LangGraph nodes consume and produce `SupportState`. QA judgements are returned
as `QAVerdict`. Document processing uses `DocumentChunk` and `RetrievalResult`.

DO NOT modify these contracts without updating all consumers.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


# ── LangGraph-Compatible Shared State ─────────────────────────────────────────

class SupportState(TypedDict):
    """Shared state object flowing through the LangGraph agent DAG.

    Every LangGraph node reads fields from this TypedDict and returns a
    *partial* dict of fields it wants to update. LangGraph merges the
    partial updates back into the overall state automatically.

    The `messages` field uses `Annotated[list, operator.add]` so that
    LangGraph appends new messages rather than overwriting the list — this
    is the standard LangGraph reducer pattern for message history.

    Human-in-the-loop fields (`requires_human_review`, `human_feedback`)
    enable LangGraph's interrupt/resume mechanism. When `requires_human_review`
    is True, the orchestrator raises `HumanReviewRequired` and LangGraph
    checkpoints the state. A human can then supply `human_feedback` and
    resume the graph from the saved checkpoint.

    Attributes:
        user_query: The raw natural-language query from the end user.
        session_id: Unique identifier for the conversation session (used
            by LangGraph's checkpointer for HITL and replay).
        intent: Classified intent label. One of VALID_INTENTS.
        retrieved_context: Ordered list of text chunks retrieved from the
            vector store, most relevant first.
        draft_reply: The initial reply drafted by DraftWriterAgent before QA.
        final_reply: The QA-approved reply to return to the user.
        retry_count: Number of draft-QA cycles completed. Reset to 0 each
            new query. Escalation triggers when retry_count >= max_retries.
        metadata: Arbitrary key-value pairs for audit, tracing, or feature
            flags. E.g. {"trace_id": "...", "escalation_reason": "..."}.
        requires_human_review: When True, the orchestrator pauses the graph
            and waits for a human to review the draft reply.
        human_feedback: Free-text feedback provided by a human reviewer.
            The DraftWriterAgent incorporates this on the next retry cycle.
        messages: Append-only list of agent step records for audit logging.
            Uses operator.add reducer so LangGraph accumulates across nodes.
    """

    user_query: str
    session_id: str
    intent: str
    retrieved_context: list[str]
    draft_reply: str
    final_reply: str
    retry_count: int
    metadata: dict
    # Human-in-the-loop fields
    requires_human_review: bool
    human_feedback: str | None
    # Append-only message history via LangGraph reducer
    messages: Annotated[list[dict], operator.add]


# ── QA Verdict ────────────────────────────────────────────────────────────────

class QAVerdict(BaseModel):
    """Structured evaluation result returned by QAAgent after reviewing a draft reply.

    When `passed` is False, the orchestrator either retries (if retry_count <
    max_retries) or escalates to a human (if retry_count >= max_retries).

    Attributes:
        passed: True if the draft reply passes all quality checks.
        issues: Human-readable list of specific problems found. Empty when passed=True.
        missing_policy_info: True if the reply omits required policy information.
        bad_tone: True if the reply uses inappropriate, rude, or unprofessional tone.
        inaccurate: True if the reply contains factual inaccuracies vs. source docs.
        confidence_score: Float in [0, 1] representing QA confidence. Used to
            prioritise which failed replies need human attention.
    """

    passed: bool = Field(..., description="True if reply passes all QA checks.")
    issues: list[str] = Field(
        default_factory=list,
        description="Specific problems found. Empty list when passed=True.",
    )
    missing_policy_info: bool = Field(
        default=False,
        description="Reply is missing required policy or procedural context.",
    )
    bad_tone: bool = Field(
        default=False,
        description="Reply contains inappropriate, rude, or unprofessional language.",
    )
    inaccurate: bool = Field(
        default=False,
        description="Reply contains factual inaccuracies relative to retrieved context.",
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="QA model confidence score. Lower scores may warrant human review.",
    )


# ── Document Types ─────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A processed, embeddable unit of a source document.

    Created by SemanticChunker after splitting raw LlamaIndex Documents.
    Stored in ChromaDB as a vector with this metadata attached.

    Attributes:
        chunk_id: UUID string assigned by LlamaIndex (TextNode.node_id).
        source_path: Absolute or relative path of the originating file.
        content: The plain-text content of this chunk.
        metadata: Arbitrary metadata from the source document (page number,
            section title, loaded_at timestamp, etc.).
        token_count: Approximate token count (computed during chunking).
    """

    chunk_id: str
    source_path: str
    content: str
    metadata: dict = field(default_factory=dict)
    token_count: int = 0


@dataclass
class RetrievalResult:
    """A single retrieved document chunk paired with its relevance score.

    Returned by HybridRetriever for each query. Used by DraftWriterAgent
    to compose the context-grounded reply.

    Attributes:
        chunk: The retrieved DocumentChunk.
        score: Relevance score in [0, 1]. Higher = more relevant.
        retrieval_method: How this chunk was retrieved: "dense" (vector
            similarity), "sparse" (BM25 keyword), or "hybrid" (RRF-combined).
    """

    chunk: DocumentChunk
    score: float
    retrieval_method: str  # "dense" | "sparse" | "hybrid"


# ── Agent Action Audit Record ──────────────────────────────────────────────────

class AgentAction(BaseModel):
    """Records a single agent step for audit logging and observability.

    Appended to `SupportState.messages` by each node before returning.

    Attributes:
        agent_name: Identifier of the agent that performed the action.
        action: Short verb describing what the agent did (e.g., "classify_intent").
        input_summary: Brief description of the agent's input.
        output_summary: Brief description of the agent's output.
        latency_ms: Wall-clock time the agent took to complete the action.
        success: False if the agent encountered a handled error.
    """

    agent_name: str
    action: str
    input_summary: str
    output_summary: str
    latency_ms: float
    success: bool


# ── Ingestion Result ───────────────────────────────────────────────────────────

class IngestionResult(BaseModel):
    """Summary of a completed ingestion pipeline run.

    Returned by IngestionPipeline.run() and surfaced via the POST /ingest endpoint.

    Attributes:
        total_documents: Number of source files discovered and attempted.
        total_chunks: Number of chunks successfully stored in ChromaDB.
        failed_documents: Paths of files that could not be loaded or chunked.
        success: True if all documents ingested without failures.
        duration_seconds: Total wall-clock time for the pipeline run.
    """

    total_documents: int
    total_chunks: int
    failed_documents: list[str] = Field(default_factory=list)
    success: bool
    duration_seconds: float


# ── Intent Label ──────────────────────────────────────────────────────────────

#: Frozenset of all valid intent labels. IntentClassifierAgent MUST return one of these.
VALID_INTENTS: frozenset[str] = frozenset(
    {
        "refund",      # Customer requesting money back
        "technical",   # Technical issue / product malfunction
        "billing",     # Invoice, payment, or subscription questions
        "general",     # General enquiry not matching other categories
        "escalate",    # Explicit escalation request or sensitive complaint
    }
)

#: Type alias for a valid intent string. One of VALID_INTENTS.
IntentLabel = str
