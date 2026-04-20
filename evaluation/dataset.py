"""
FILE: customer_support_rag/evaluation/dataset.py
=================================================
Dataset loading and validation for the RAG evaluation pipeline.

Responsible for:
    - Loading the golden evaluation dataset from JSON
    - Strongly validating the schema of each sample
    - Converting raw dicts into typed EvalSample objects
"""

from __future__ import annotations

import json
from pathlib import Path

from core.types import EvalSample, VALID_INTENTS
from core.logging import get_logger

logger = get_logger(__name__)


def load_eval_dataset(path: str | Path) -> list[EvalSample]:
    """Load evaluation dataset from a structured JSON file.

    Accepts two formats:
        - A top-level JSON list of sample dicts
        - A dict with a ``"samples"`` key containing the list

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file or any sample fails schema validation.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Allow direct list or { "samples": [...] }
    if isinstance(data, list):
        samples_data = data
    elif isinstance(data, dict):
        if "samples" not in data:
            raise ValueError("Invalid JSON format: dict missing 'samples' field.")
        samples_data = data["samples"]
    else:
        raise ValueError("Invalid JSON format: expected list or dict at top level.")

    if not isinstance(samples_data, list):
        raise ValueError("'samples' must be a list.")

    samples: list[EvalSample] = []

    for idx, item in enumerate(samples_data):
        if not isinstance(item, dict):
            raise ValueError(f"Sample at index {idx} is not a dictionary.")

        # Strongly type-check required keys
        question = item.get("question")
        ground_truth = item.get("ground_truth")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Sample at index {idx} misses valid 'question' (str).")
        if not isinstance(ground_truth, str) or not ground_truth.strip():
            raise ValueError(f"Sample at index {idx} misses valid 'ground_truth' (str).")

        # Validate contexts
        ground_truth_contexts = item.get("ground_truth_contexts", [])
        if not isinstance(ground_truth_contexts, list):
            raise ValueError(
                f"Sample at index {idx} 'ground_truth_contexts' must be list of str."
            )

        # Validate expected_intent
        expected_intent = item.get("expected_intent")
        if expected_intent is not None:
            if not isinstance(expected_intent, str) or expected_intent not in VALID_INTENTS:
                raise ValueError(
                    f"Sample at index {idx} 'expected_intent' {expected_intent!r} "
                    f"is invalid. Must be one of {list(VALID_INTENTS)}."
                )

        # Validate metadata
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Sample at index {idx} 'metadata' must be dictionary.")

        samples.append(
            EvalSample(
                question=question,
                ground_truth=ground_truth,
                ground_truth_contexts=ground_truth_contexts,
                expected_answer_points=item.get("expected_answer_points", []),
                expected_intent=expected_intent,
                metadata=metadata,
            )
        )

    logger.info("dataset.loaded", num_samples=len(samples), path=str(path))
    return samples
