"""
FILE: customer_support_rag/evaluation/__init__.py
==================================================
Public API for the evaluation package.

Import the key functions from sub-modules here to allow clean external usage:

    from evaluation import load_eval_dataset, run_evaluation
"""

from evaluation.dataset import load_eval_dataset
from evaluation.runner import run_all_samples, run_system_on_sample
from evaluation.ragas_eval import (
    build_ragas_dataset,
    run_ragas_evaluation,
    compute_intent_accuracy,
)
from evaluation.langsmith_tracker import get_langsmith_client, push_results_to_langsmith
from evaluation.reporting import print_summary, save_detailed_results
from evaluation.evaluate import run_evaluation

__all__ = [
    "load_eval_dataset",
    "run_system_on_sample",
    "run_all_samples",
    "build_ragas_dataset",
    "run_ragas_evaluation",
    "compute_intent_accuracy",
    "get_langsmith_client",
    "push_results_to_langsmith",
    "print_summary",
    "save_detailed_results",
    "run_evaluation",
]
