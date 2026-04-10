# FILE: customer_support_rag/api/main.py
"""FastAPI application factory for the Multi-Agent RAG Customer Support System.

Responsibilities:
    - Call configure_logging() exactly once at startup via lifespan context.
    - Apply CORS middleware (permissive in dev; lock down in prod via Settings).
    - Register a global exception handler that converts APIError → HTTP response.
    - Mount the /chat and /ingest routers.

Usage:
    uvicorn api.main:app --reload --port 8000

Or via Makefile:
    make dev
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from core.exceptions import APIError
from core.logging import configure_logging, get_logger

logger: structlog.BoundLogger = get_logger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Async lifespan context manager — runs startup logic before yield.

    Replaces the deprecated @app.on_event("startup") pattern.
    configure_logging() is idempotent, so calling it here is safe even if
    structlog was already touched during import time.
    """
    configure_logging()
    logger.info("api.startup", message="Multi-Agent RAG API starting up")
    yield
    logger.info("api.shutdown", message="Multi-Agent RAG API shutting down")


# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Construct and return the configured FastAPI application.

    Separated from module-level instantiation so the app can be recreated
    in tests with different settings via dependency injection overrides.

    Returns:
        A fully configured FastAPI instance with middleware and routers.
    """
    app = FastAPI(
        title="Customer Support RAG API",
        description=(
            "Multi-Agent Retrieval-Augmented Generation system for customer support. "
            "Routes: POST /chat, POST /chat/resume, POST /ingest, GET /ingest/status/{job_id}, GET /health."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Allow all origins in development. In production, restrict to your domain.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── X-Request-ID middleware ───────────────────────────────────────────────
    # Generates a unique request ID for each request (or reuses the client's
    # X-Request-ID header if provided). Binds it to the structlog context so it
    # appears in every log line, and returns it in the response headers for
    # end-to-end tracing across services.
    class RequestIdMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
            request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
            # Bind to structlog context for the duration of this request
            structlog.contextvars.bind_contextvars(request_id=request_id)
            try:
                response: Response = await call_next(request)
            finally:
                structlog.contextvars.unbind_contextvars("request_id")
            response.headers["X-Request-ID"] = request_id
            return response

    app.add_middleware(RequestIdMiddleware)

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        """Translate APIError into a structured JSON HTTP response.

        All APIError subclasses carry a `status_code` attribute set in __init__.
        This handler extracts it and returns a consistent error envelope.
        """
        logger.warning(
            "api.error",
            path=str(request.url),
            status_code=exc.status_code,
            message=str(exc),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": str(exc), "status_code": exc.status_code},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    from api.routes.chat import router as chat_router
    from api.routes.ingest import router as ingest_router

    app.include_router(chat_router)
    app.include_router(ingest_router)

    return app


# Module-level app instance — imported by uvicorn and tests.
app: FastAPI = create_app()
