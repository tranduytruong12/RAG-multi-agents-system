# FILE: customer_support_rag/core/exceptions.py
"""Custom exception hierarchy for the Multi-Agent RAG Customer Support System.

All exceptions inherit from RAGBaseException, which attaches a `context` dict
for structured error metadata. Use the context dict to pass machine-readable
data to the structlog logger:

    try:
        ...
    except LoaderError as e:
        logger.error("loader_failed", error=str(e), **e.context)
        raise
"""

from __future__ import annotations


class RAGBaseException(Exception):
    """Root exception for all errors originating in this system.

    Args:
        message: Human-readable description of what went wrong.
        context: Optional dict of machine-readable metadata for structured logging.

    Attributes:
        context: Structured metadata attached to this exception instance.
    """

    def __init__(self, message: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.context: dict = context or {}


# ── Ingestion Exceptions ───────────────────────────────────────────────────────

class IngestionError(RAGBaseException):
    """Raised when the ingestion pipeline encounters an unrecoverable error.

    This is the base class for all ingestion-related failures. Catch this
    class to handle any ingestion failure without caring about the specific
    sub-phase (loading vs. chunking).
    """


class LoaderError(IngestionError):
    """Raised when a document loader fails to read or parse a file.

    Expected context keys:
        file_path (str): Path of the file that failed to load.
        file_type (str): "markdown" or "pdf".
        original_error (str): str(original_exception).
    """


class ChunkerError(IngestionError):
    """Raised when document chunking fails for one or more documents.

    Expected context keys:
        doc_count (int): Number of documents being chunked when failure occurred.
        original_error (str): str(original_exception).
    """


# ── Retrieval Exceptions ───────────────────────────────────────────────────────

class RetrievalError(RAGBaseException):
    """Raised when retrieval from the vector store fails.

    This is the base class for all retrieval-related failures. Catch this
    to handle any retrieval failure without caring about the sub-cause.
    """


class VectorStoreError(RetrievalError):
    """Raised on ChromaDB connection failures, collection not found, or query errors.

    Expected context keys:
        host (str): ChromaDB host that was targeted.
        port (int): ChromaDB port.
        collection (str): Collection name being accessed.
        operation (str): "connect", "add", "query", or "delete".
        original_error (str): str(original_exception).
    """


class EmbeddingError(RetrievalError):
    """Raised when the embedding model fails to generate vectors.

    Expected context keys:
        model_name (str): Embedding model that was called.
        input_length (int): Character length of the text being embedded.
        original_error (str): str(original_exception).
    """


# ── Agent Exceptions ───────────────────────────────────────────────────────────

class AgentError(RAGBaseException):
    """Raised when a LangGraph agent node encounters an unrecoverable error.

    This is the base class for all agent-layer failures. The orchestrator
    catches this to decide whether to retry or escalate.
    """


class IntentClassificationError(AgentError):
    """Raised by IntentClassifierAgent when intent cannot be determined.

    This may occur if the LLM returns a label not in VALID_INTENTS, or if
    the LLM call itself fails after all retries.

    Expected context keys:
        user_query (str): The query that failed classification.
        llm_response (str): Raw LLM output that was not a valid intent.
    """


class QAFailureError(AgentError):
    """Raised by OrchestratorAgent when QA fails after the maximum retry count.

    This triggers either a graceful escalation reply or human-in-the-loop review.

    Expected context keys:
        retry_count (int): Number of retries attempted.
        max_retries (int): Configured maximum from settings.
        final_issues (list[str]): QAVerdict.issues from the last failed attempt.
    """


class HumanReviewRequired(AgentError):
    """Raised to signal that the current draft requires a human reviewer.

    LangGraph catches this exception at the orchestrator level and
    checkpoints the state, enabling interrupt/resume HITL flow.

    Args:
        message: Description of why human review is needed.
        state_snapshot: Copy of the relevant SupportState fields at the
            time of escalation. Used by the human reviewer interface.

    Attributes:
        state_snapshot: Frozen dict of relevant state fields.
    """

    def __init__(self, message: str, state_snapshot: dict | None = None) -> None:
        super().__init__(message)
        self.state_snapshot: dict = state_snapshot or {}


# ── API Exceptions ─────────────────────────────────────────────────────────────

class APIError(RAGBaseException):
    """Raised for errors that originate in or are surfaced at the FastAPI layer.

    The FastAPI exception handler translates these into HTTP responses using
    the `status_code` attribute.

    Args:
        message: Human-readable error description.
        status_code: HTTP status code to return. Defaults to 500.

    Attributes:
        status_code: HTTP status code for the error response.
    """

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message, context={"status_code": status_code})
        self.status_code: int = status_code
