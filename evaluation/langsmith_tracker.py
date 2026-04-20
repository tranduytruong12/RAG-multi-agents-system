"""
FILE: customer_support_rag/evaluation/langsmith_tracker.py
===========================================================
LangSmith dataset push and per-run feedback logging.

Responsible for:
    - Creating or reusing a LangSmith **golden dataset** (static, deduplicated)
    - Creating a **run-specific experiment tag** so each eval run is isolated
    - Attaching per-sample RAGAS scores as run feedback
    - Attaching intent correctness as a boolean feedback signal
    - Gracefully handling auth failures (expired / invalid API key)

Run isolation strategy in LangSmith:
    The *dataset* is treated as a static golden truth — examples are only added
    when a question is NOT already present (deduplication by question text).
    Each run is distinguished by ``eval_run_id`` stored in:
        - ``metadata["eval_run_id"]``  on every example it creates
        - The ``LANGCHAIN_TAGS`` env variable (picked up by @traceable decorators
          in runner.py so every trace is tagged with the run ID)

Environment variables required:
    LANGCHAIN_API_KEY      — LangSmith API key (lsv2_pt_...)
    LANGCHAIN_TRACING_V2   — must be "true"
    LANGCHAIN_PROJECT      — project name (overridden by --project-name CLI arg)
"""

from __future__ import annotations

import os

from langsmith import Client as LangSmithClient

from config.settings import settings
from core.types import EvalRow
from core.logging import get_logger

logger = get_logger(__name__)


def get_langsmith_client() -> LangSmithClient | None:
    """Attempt to build a LangSmith client.

    Returns None if construction fails (e.g. missing env vars).
    Note: The client does NOT validate the API key on construction —
    auth errors are only raised on first API call.
    """
    # Prefer LANGCHAIN_* first because this repo stores LangSmith config that way.
    api_key = settings.langchain_api_key or settings.langsmith_api_key
    api_url = settings.langchain_endpoint or settings.langsmith_endpoint or None

    if not api_key:
        logger.warning(
            "langsmith.client_missing_api_key",
            msg="Skipping LangSmith push — LangSmith API key is missing.",
        )
        return None

    try:
        client = LangSmithClient(api_key=api_key, api_url=api_url)
        logger.info(
            "langsmith.client_created",
            api_url=api_url or "https://api.smith.langchain.com",
        )
        return client
    except BaseException as exc:
        logger.warning("langsmith.client_failed", error=str(exc))
        return None


def set_run_tags(eval_run_id: str) -> None:
    """Inject ``eval_run_id`` into ``LANGCHAIN_TAGS`` so all traces in this
    run are automatically tagged.

    LangSmith's ``@traceable`` decorator reads ``LANGCHAIN_TAGS`` at trace
    creation time, making traces from different runs easy to filter in the UI.
    """
    existing = os.environ.get("LANGCHAIN_TAGS", "")
    tags = {t for t in existing.split(",") if t} | {eval_run_id}
    os.environ["LANGCHAIN_TAGS"] = ",".join(sorted(tags))
    logger.info("langsmith.tags_set", tags=os.environ["LANGCHAIN_TAGS"])


def _get_or_create_dataset(client: LangSmithClient, dataset_name: str):
    """Return the named LangSmith dataset, creating it if it does not exist.

    Returns None and logs a warning if any API error occurs (e.g. 401).
    """
    try:
        ds = client.read_dataset(dataset_name=dataset_name)
        logger.info("langsmith.dataset_found", id=ds.id, name=dataset_name)
        return ds
    except Exception:
        pass

    logger.info("langsmith.creating_dataset", name=dataset_name)
    try:
        ds = client.create_dataset(
            dataset_name=dataset_name,
            description="Golden eval dataset for customer-support RAG",
        )
        logger.info("langsmith.dataset_created", id=ds.id)
        return ds
    except Exception as exc:
        logger.warning(
            "langsmith.dataset_create_failed",
            error=str(exc),
            msg="Skipping LangSmith push — check LANGCHAIN_API_KEY.",
        )
        return None


def _fetch_existing_questions(client: LangSmithClient, dataset_id: str) -> set[str]:
    """Return the set of question strings already present in the dataset.

    Used to avoid adding duplicate examples on repeated eval runs.
    """
    try:
        existing = {
            ex.inputs.get("question", "")
            for ex in client.list_examples(dataset_id=dataset_id)
        }
        logger.info("langsmith.existing_examples_fetched", count=len(existing))
        return existing
    except Exception as exc:
        logger.warning("langsmith.list_examples_failed", error=str(exc))
        return set()


def push_results_to_langsmith(
    client: LangSmithClient,
    dataset_name: str,
    rows: list[EvalRow],
    detailed_ragas_scores: list[dict],
    eval_run_id: str = "unknown-run",
) -> None:
    """Push evaluation results to LangSmith as a dataset + per-run feedback.

    Run isolation:
        - The **dataset** is treated as a static golden set. Examples are
          only added if their ``question`` is not already present
          (deduplication). Re-running with identical data will NOT grow the
          dataset.
        - Each **run** is tagged via ``eval_run_id`` stored as example
          metadata and via ``LANGCHAIN_TAGS`` on all traces.

    Steps performed:
        1. Get or create the named dataset.
        2. Fetch existing questions to deduplicate.
        3. For NEW questions only: create a golden example.
        4. For ALL rows: if a ``langsmith_run_id`` exists, attach RAGAS
           metric scores and intent correctness as run feedback.

    Args:
        client:               Authenticated LangSmith client.
        dataset_name:         Target Dataset name. Created if absent.
        rows:                 Evaluated EvalRow instances.
        detailed_ragas_scores: Per-sample metric dicts from RAGAS.
        eval_run_id:          Unique ID for this evaluation run.

    Raises:
        Does NOT raise — all errors are caught and logged as warnings.
    """
    # ── Step 1: get or create dataset ─────────────────────────────────────────
    ds = _get_or_create_dataset(client, dataset_name)
    if ds is None:
        return  # auth or network error already logged

    # ── Step 2: fetch existing questions (for deduplication) ──────────────────
    existing_questions = _fetch_existing_questions(client, str(ds.id))
    new_count = skipped_count = 0

    # ── Steps 3 & 4: push examples + attach feedback ─────────────────────────
    for row, metric_scores in zip(rows, detailed_ragas_scores):

        # ── 3. Only add NEW examples to the golden dataset ───────────────────
        if row.question not in existing_questions:
            try:
                client.create_example(
                    inputs={"question": row.question},
                    outputs={"ground_truth": row.ground_truth},
                    metadata={
                        "predicted_answer":        row.answer,
                        "retrieved_context_count": len(row.contexts),
                        "intent":                  row.intent,
                        "retry_count":             row.retry_count,
                        "eval_run_id":             eval_run_id,   # tag the run
                    },
                    dataset_id=ds.id,
                )
                existing_questions.add(row.question)  # prevent in-batch duplicates
                new_count += 1
            except Exception as example_err:
                logger.warning(
                    "langsmith.example_create_failed",
                    session_id=row.session_id,
                    error=str(example_err),
                )
        else:
            skipped_count += 1

        # ── 4. Attach feedback to this run's trace (always, not just new) ────
        if row.langsmith_run_id:
            _attach_feedback(client, row, metric_scores, eval_run_id)

    logger.info(
        "langsmith.push_complete",
        dataset=dataset_name,
        new_examples=new_count,
        skipped_duplicates=skipped_count,
        eval_run_id=eval_run_id,
    )


def _attach_feedback(
    client: LangSmithClient,
    row: EvalRow,
    metric_scores: dict,
    eval_run_id: str,
) -> None:
    """Attach RAGAS scores, intent correctness, and run ID tag to a trace."""
    for metric, score in metric_scores.items():
        if score is None:
            continue
        try:
            client.create_feedback(
                run_id=row.langsmith_run_id,
                key=metric,
                score=score,
                comment=f"eval_run_id={eval_run_id}",
            )
        except Exception as fb_err:
            logger.warning(
                "langsmith.feedback_failed",
                metric=metric,
                run_id=row.langsmith_run_id,
                error=str(fb_err),
            )

    # Intent correctness as a boolean signal (1.0 = correct, 0.0 = wrong)
    try:
        client.create_feedback(
            run_id=row.langsmith_run_id,
            key="intent_correct",
            score=1.0 if row.intent == row.expected_intent else 0.0,
            comment=f"eval_run_id={eval_run_id}",
        )
    except Exception as intent_err:
        logger.warning(
            "langsmith.intent_feedback_failed",
            run_id=row.langsmith_run_id,
            error=str(intent_err),
        )
