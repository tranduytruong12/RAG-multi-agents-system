"""
FILE: customer_support_rag/evaluation/reporting.py
===================================================
Results reporting for the RAG evaluation pipeline.

Responsible for:
    - Pretty-printing aggregated metrics as a terminal table
    - Saving aggregated metrics to a run-specific subdirectory
      ``{output_dir}/{eval_run_id}/metrics_summary.json``
    - Saving per-sample diagnostic data to ``detailed_results.json``

Run isolation strategy:
    Each call to print_summary / save_detailed_results receives an
    ``eval_run_id`` string (e.g. ``eval-3f7a1b2c``) and writes output into
    ``{output_dir}/{eval_run_id}/`` so repeated runs never overwrite each other.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.types import EvalRow
from core.logging import get_logger

logger = get_logger(__name__)


def get_run_output_dir(base_output_dir: str | Path, eval_run_id: str) -> Path:
    """Return (and create) a run-specific output subdirectory.

    Example::

        get_run_output_dir("evaluation/results", "eval-3f7a1b2c")
        # → PosixPath('evaluation/results/eval-3f7a1b2c')
    """
    run_dir = Path(base_output_dir) / eval_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def print_summary(
    ragas_scores: dict[str, float],
    intent_accuracy: float,
    output_dir: str | Path,
    eval_run_id: str = "run",
) -> None:
    """Pretty-print aggregated evaluation results as a box table and save JSON.

    Results are saved to ``{output_dir}/{eval_run_id}/metrics_summary.json``
    so repeated runs with the same ``output_dir`` never overwrite each other.

    Args:
        ragas_scores:    Mean RAGAS metric scores keyed by metric name.
        intent_accuracy: Intent classification accuracy in [0.0, 1.0].
        output_dir:      Base directory for all evaluation runs.
        eval_run_id:     Unique run identifier used as the subdirectory name.
    """
    run_dir = get_run_output_dir(output_dir, eval_run_id)

    results = {**ragas_scores, "intent_accuracy": intent_accuracy}

    # ── Terminal box table ─────────────────────────────────────────────────────
    title = "RAG Evaluation Results"
    key_width  = max(len(k) for k in results.keys()) + 2
    val_width  = 10
    total_width = key_width + val_width + 5

    def rule(char: str = "═") -> str:
        return char * total_width

    print(f"\n╔{rule()}╗")
    print(f"║{title:^{total_width}}║")
    print(f"╠{rule()}╣")
    for key, value in results.items():
        print(f"║  {key:<{key_width - 2}}│  {value:>{val_width}.4f}  ║")
    print(f"╚{rule()}╝\n")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    output_path = run_dir / "metrics_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Metrics summary saved to: {output_path}")
    logger.info("reporting.summary_saved", path=str(output_path), scores=results)


def save_detailed_results(
    rows: list[EvalRow],
    detailed_scores: list[dict],
    output_dir: str | Path,
    eval_run_id: str = "run",
) -> None:
    """Save per-sample diagnostic data to ``{output_dir}/{eval_run_id}/detailed_results.json``.

    Each record contains the question, answer, contexts, ground truth, intent
    classification result, session metadata, and per-metric RAGAS scores.

    Args:
        rows:            EvalRow objects from the runner.
        detailed_scores: Per-sample RAGAS metric dicts.
        output_dir:      Base directory for all evaluation runs.
        eval_run_id:     Unique run identifier used as the subdirectory name.
    """
    run_dir = get_run_output_dir(output_dir, eval_run_id)

    data = [
        {
            "question":         r.question,
            "answer":           r.answer,
            "contexts":         r.contexts,
            "ground_truth":     r.ground_truth,
            "intent":           r.intent,
            "expected_intent":  r.expected_intent,
            "retry_count":      r.retry_count,
            "session_id":       r.session_id,
            "langsmith_run_id": r.langsmith_run_id,
            "ragas_scores":     scores,
        }
        for r, scores in zip(rows, detailed_scores)
    ]

    output_path = run_dir / "detailed_results.json"
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    print(f"Detailed diagnostics saved to: {output_path}")
    logger.info("reporting.detailed_saved", path=str(output_path), num_rows=len(data))
