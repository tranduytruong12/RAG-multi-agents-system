# FILE: customer_support_rag/core/logging.py
"""Structured logging configuration using structlog.

Call `configure_logging()` exactly once at application startup (in api/main.py
lifespan or at the top of any CLI entry point). Then use `get_logger(__name__)`
to obtain a bound logger in every module.

Log format is controlled by `settings.log_format`:
    - "json"    → JSONRenderer (production / Docker / log aggregators)
    - "console" → ConsoleRenderer with colors (local development)

Example:
    from core.logging import get_logger

    logger = get_logger(__name__)

    async def ingest(path: str) -> None:
        logger.info("ingestion_started", source_path=path)
        # ... work ...
        logger.info("ingestion_complete", chunks=42, duration_ms=310)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from config.settings import settings


def configure_logging() -> None:
    """Configure structlog and the standard-library logging root logger.

    Must be called exactly once at application startup. Calling it multiple
    times is safe (structlog is idempotent) but wasteful.

    The configuration:
        1. Merges contextvars bindings (set via `structlog.contextvars.bind_contextvars`)
           into every log record — used to attach `request_id`, `session_id`, etc.
        2. Renders with JSONRenderer in production or ConsoleRenderer in development.
        3. Bridges stdlib `logging` through structlog so third-party libraries
           (LlamaIndex, LangGraph, ChromaDB) emit structured records.
    """
    shared_processors: list[Any] = [
        # Merge contextvars (request_id, session_id, agent_name, etc.)
        structlog.contextvars.merge_contextvars,
        # Inject log level string ("info", "error", etc.)
        structlog.stdlib.add_log_level,
        # Inject the logger name (__name__ of the calling module)
        structlog.stdlib.add_logger_name,
        # ISO 8601 timestamp
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Extract and format stack traces for exception events
        structlog.processors.StackInfoRenderer(),
        # Format exc_info into a structured "exception" key
        structlog.processors.format_exc_info,
    ]

    if settings.log_format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True, exception_formatter=structlog.dev.plain_traceback)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib logging → structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Remove any existing handlers to avoid duplicate output
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(
        getattr(logging, settings.log_level.upper(), logging.INFO)
    )

    # Silence overly verbose third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger for the given module name.

    This is the primary factory function for obtaining loggers throughout
    the codebase. Always pass `__name__` so logs include the module path.

    Args:
        name: Logger name, typically `__name__` of the calling module.

    Returns:
        A structlog BoundLogger that emits structured events as either
        JSON or colored console output depending on `settings.log_format`.

    Example:
        logger = get_logger(__name__)
        logger.info("retrieval_complete", query="refund policy", top_k=5, latency_ms=42.3)
        logger.error("vector_store_error", host="localhost", port=8001, error="Connection refused")
    """
    return structlog.stdlib.get_logger(name)


def bind_request_context(
    request_id: str,
    session_id: str | None = None,
    agent_name: str | None = None,
) -> None:
    """Bind request-scoped context variables to all log records in this async context.

    Call this at the start of each API request handler. All subsequent
    `logger.*()` calls in the same async context will automatically include
    these fields without needing to pass them explicitly.

    Args:
        request_id: Unique identifier for the HTTP request (e.g., UUID).
        session_id: Optional conversation session identifier.
        agent_name: Optional name of the currently active agent node.

    Example:
        @app.post("/chat")
        async def chat(request: ChatRequest) -> ChatResponse:
            bind_request_context(
                request_id=str(uuid4()),
                session_id=request.session_id,
            )
            # All logs below automatically include request_id and session_id
            result = await orchestrator.run(state)
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        **({"session_id": session_id} if session_id else {}),
        **({"agent_name": agent_name} if agent_name else {}),
    )
