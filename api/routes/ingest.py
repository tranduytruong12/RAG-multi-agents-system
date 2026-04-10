# FILE: customer_support_rag/api/routes/ingest.py
"""Ingest router: POST /ingest, GET /ingest/status/{job_id}.

POST /ingest
    Accepts an optional `source_dir` path. Runs IngestionPipeline.run() as a
    FastAPI BackgroundTask. Returns immediately with a job_id for polling.

GET /ingest/status/{job_id}
    Returns the current status of the background ingestion job using a
    simple in-memory dict. Job states: "pending" → "running" → "done" | "failed".

Design notes:
    - In-memory job store is sufficient for single-process deployments.
    - For multi-process/multi-replica, replace with Redis or a DB-backed store.
    - The IngestionPipeline may fail per-file without raising — check result.success.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.exceptions import APIError
from core.logging import get_logger
from core.types import IngestionResult

logger: structlog.BoundLogger = get_logger(__name__)

router = APIRouter(tags=["ingest"])

# Simple in-memory job registry.
# Key: job_id (UUID str)
# Value: {"status": "pending"|"running"|"done"|"failed", "result": ...}
_jobs: dict[str, dict[str, Any]] = {}

# Default source directory if caller doesn't specify one.
_DEFAULT_SOURCE_DIR = Path("data/docs")


# ── Request / Response models ──────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Input schema for POST /ingest."""

    source_dir: str = Field(
        default=str(_DEFAULT_SOURCE_DIR),
        description="Path to the directory containing source documents (.md, .pdf, .txt).",
    )


class IngestAcceptedResponse(BaseModel):
    """Response returned immediately when the ingest job is accepted."""

    job_id: str
    status: str = Field(default="pending")
    message: str = Field(
        default="Ingestion job accepted. Poll GET /ingest/status/{job_id} for progress."
    )


# ── Background task ───────────────────────────────────────────────────────────

async def _run_ingestion(job_id: str, source_dir: Path, reset_collection: bool) -> None:
    """Execute the IngestionPipeline and update the in-memory job store.

    This coroutine is registered as a FastAPI BackgroundTask and runs after
    the HTTP response is sent to the client.

    Args:
        job_id: The UUID string identifying this job in _jobs.
        source_dir: Root directory containing documents to ingest.
        reset_collection: When True the ChromaDB collection is wiped before
            ingestion (used for the default source_dir to prevent duplicates).
            When False, new chunks are appended to the existing collection.
    """
    _jobs[job_id]["status"] = "running"
    logger.info(
        "ingest.job.started",
        job_id=job_id,
        source_dir=str(source_dir),
        reset_collection=reset_collection,
    )

    try:
        from ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline()
        result: IngestionResult = await pipeline.run(
            source_dir,
            reset_collection=reset_collection,
        )

        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["result"] = result.model_dump()
        logger.info(
            "ingest.job.done",
            job_id=job_id,
            total_chunks=result.total_chunks,
            duration_s=result.duration_seconds,
        )

        # Invalidate the retriever cache if the collection was reset so the
        # next chat request rebuilds the index against the new collection UUID.
        if reset_collection:
            try:
                from api.routes.chat import get_retriever
                get_retriever().invalidate()
                logger.info("ingest.retriever.invalidated", job_id=job_id)
            except Exception as inv_exc:
                # Non-fatal: orchestrator may not be initialized yet on first run.
                logger.warning("ingest.retriever.invalidate.skipped", reason=str(inv_exc))

    except Exception as exc:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(exc)
        logger.error("ingest.job.failed", job_id=job_id, error=str(exc))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    status_code=202,
    summary="Trigger document ingestion",
    responses={
        202: {"description": "Job accepted and running in background."},
        422: {"description": "Validation error."},
    },
)
async def ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestAcceptedResponse:
    """Accept an ingestion request and run the pipeline as a background task.

    The pipeline discovers and ingests .md, .txt, and .pdf files from
    `source_dir`, chunks them, embeds them, and stores them in ChromaDB.

    Args:
        request: IngestRequest with optional source_dir override.
        background_tasks: FastAPI BackgroundTasks injected by the framework.

    Returns:
        IngestAcceptedResponse with job_id for polling via GET /ingest/status.
    """
    job_id = str(uuid.uuid4())
    source_dir = Path(request.source_dir)

    # ── Reset logic ────────────────────────────────────────────────────────────
    # When the caller uses the default source directory (data/docs), we wipe the
    # ChromaDB collection before re-ingesting so the same documents cannot be
    # duplicated across multiple POST /ingest calls.
    #
    # When a custom source_dir is provided (e.g. from the Streamlit sidebar), we
    # append nodes to the existing collection so previously ingested data is
    # preserved and the new directory is merged in.
    reset_collection: bool = (source_dir == _DEFAULT_SOURCE_DIR)

    _jobs[job_id] = {
        "status": "pending",
        "source_dir": str(source_dir),
        "reset_collection": reset_collection,
    }
    logger.info(
        "ingest.job.accepted",
        job_id=job_id,
        source_dir=str(source_dir),
        reset_collection=reset_collection,
    )

    background_tasks.add_task(_run_ingestion, job_id, source_dir, reset_collection)

    return IngestAcceptedResponse(job_id=job_id)


@router.get(
    "/ingest/status/{job_id}",
    summary="Get ingestion job status",
    responses={
        200: {"description": "Job status returned."},
        404: {"description": "Job ID not found."},
    },
)
async def ingest_status(job_id: str) -> JSONResponse:
    """Return the current status of a background ingestion job.

    Args:
        job_id: The UUID returned by POST /ingest.

    Returns:
        JSONResponse with job status dict: status, result (if done), error (if failed).

    Raises:
        APIError: 404 if the job_id is not recognised.
    """
    if job_id not in _jobs:
        raise APIError(f"Job not found: {job_id}", status_code=404)

    return JSONResponse(content={"job_id": job_id, **_jobs[job_id]})
