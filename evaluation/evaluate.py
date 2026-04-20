"""
FILE: customer_support_rag/evaluation/evaluate.py
==================================================
End-to-end evaluation orchestrator for the Multi-Agent RAG Customer Support system.

This file wires together the evaluation pipeline sub-modules:

    dataset.py           → load & validate the golden eval dataset
    runner.py            → invoke the OrchestratorAgent on each sample
    ragas_eval.py        → RAGAS retrieval + generation quality metrics
    langsmith_tracker.py → push results to LangSmith (optional)
    reporting.py         → print terminal table + save JSON files

Usage (from project root, with .env loaded):
    python -m evaluation.evaluate --dataset data/eval_dataset.json \\
                                  --dataset-name "v1-eval" \\
                                  --project-name "rag-eval-run-1"

Environment variables required (see .env):
    OPENAI_API_KEY         — used by RAGAS LLM/embeddings
    LANGCHAIN_API_KEY      — LangSmith API key (optional; skipped if invalid)
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_PROJECT      — LangSmith project (overridden by --project-name)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

# Make sure the project root is on sys.path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.orchestrator import OrchestratorAgent
from core.logging import get_logger
from ingestion.pipeline import IngestionPipeline

from evaluation.dataset import load_eval_dataset
from evaluation.runner import run_all_samples
from evaluation.ragas_eval import build_ragas_dataset, run_ragas_evaluation, compute_intent_accuracy
from evaluation.langsmith_tracker import get_langsmith_client, push_results_to_langsmith, set_run_tags
from evaluation.reporting import print_summary, save_detailed_results

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

async def run_evaluation(
    dataset_path: str | None,
    langsmith_dataset_name: str,
    langsmith_project: str,
    output_dir: str,
) -> dict[str, float]:
    """Full evaluation pipeline: load → ingest → run system → RAGAS → LangSmith → report.

    Steps:
        1. Load and validate the golden dataset.
        2. Run the ingestion pipeline to populate the local vector DB.
        3. Invoke the OrchestratorAgent on every sample.
        4. Compute RAGAS metrics and intent accuracy.
        5. (Optional) Push results to LangSmith.
        6. Save results to disk and print summary table.

    Returns:
        Combined scores dict: {metric_name: float, "intent_accuracy": float}
    """
    os.environ["LANGCHAIN_PROJECT"] = langsmith_project

    eval_run_id = f"eval-{uuid.uuid4().hex[:8]}"
    logger.info("eval.start", run_id=eval_run_id, project=langsmith_project)

    # Tag all LangSmith traces in this run with eval_run_id
    set_run_tags(eval_run_id)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    samples = load_eval_dataset(dataset_path) if dataset_path else []
    if not samples:
        raise ValueError("Evaluation requires a valid non-empty dataset.")
    logger.info("eval.dataset_loaded", num_samples=len(samples))

    # ── 2. Ingest documents ───────────────────────────────────────────────────
    # logger.info("eval.auto_ingest", msg="Populating local vector DB...")
    # ingest_pipeline = IngestionPipeline()
    # await ingest_pipeline.run(Path("data/docs"), reset_collection=True)

    # ── 3. Run agent on every sample ──────────────────────────────────────────
    orchestrator = OrchestratorAgent()
    rows = await run_all_samples(orchestrator, samples, eval_run_id)
    logger.info("eval.runs_complete", num_rows=len(rows))

    # ── 4. Compute metrics ────────────────────────────────────────────────────
    intent_accuracy = compute_intent_accuracy(rows)
    ragas_dataset = build_ragas_dataset(rows)
    mean_scores, detailed_scores = run_ragas_evaluation(ragas_dataset)

    # ── 5. Push to LangSmith (optional) ──────────────────────────────────────
    langsmith_client = get_langsmith_client()
    if langsmith_client:
        try:
            push_results_to_langsmith(
                langsmith_client, langsmith_dataset_name, rows, detailed_scores,
                eval_run_id=eval_run_id,
            )
        except Exception as ls_err:
            logger.warning(
                "eval.langsmith_push_failed",
                error=str(ls_err),
                msg="LangSmith push skipped. Results are still saved locally.",
            )

    # ── 6. Save and report ────────────────────────────────────────────────────
    print_summary(mean_scores, intent_accuracy, output_dir, eval_run_id=eval_run_id)
    save_detailed_results(rows, detailed_scores, output_dir, eval_run_id=eval_run_id)

    combined = {**mean_scores, "intent_accuracy": intent_accuracy}
    logger.info("eval.done", run_id=eval_run_id, scores=combined)
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Multi-Agent RAG system with LangSmith + RAGAS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the golden evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--dataset-name",
        default="rag-eval-dataset",
        help="LangSmith dataset name to create / update.",
    )
    parser.add_argument(
        "--project-name",
        default=os.getenv("LANGCHAIN_PROJECT", "rag-eval"),
        help="LangSmith project name for this evaluation run.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/results",
        help="Directory to write JSON results files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        run_evaluation(
            dataset_path=args.dataset,
            langsmith_dataset_name=args.dataset_name,
            langsmith_project=args.project_name,
            output_dir=args.output_dir,
        )
    )
