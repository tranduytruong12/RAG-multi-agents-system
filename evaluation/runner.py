"""
FILE: customer_support_rag/evaluation/runner.py
================================================
Agent runner for the RAG evaluation pipeline.

Responsible for:
    - Invoking OrchestratorAgent on each EvalSample
    - Capturing LangSmith trace run IDs per sample
    - Collecting outputs into EvalRow objects
    - Handling per-sample failures gracefully (logs warning, skips)
"""

from __future__ import annotations

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from agents.orchestrator import OrchestratorAgent
from core.types import SupportState, EvalSample, EvalRow
from core.logging import get_logger

logger = get_logger(__name__)


@traceable(name="Evaluate Single Sample")
async def run_system_on_sample(
    orchestrator: OrchestratorAgent,
    sample: EvalSample,
    session_id: str,
) -> EvalRow:
    """Invoke the OrchestratorAgent on a single EvalSample and collect outputs.

    Decorated with ``@traceable`` so LangSmith records the full execution tree
    for this sample. The run ID is captured immediately and stored on the
    returned EvalRow so RAGAS feedback can be attached to the exact trace.
    """
    # Grab the run ID dynamically so we can attach feedback strictly to THIS run
    run_tree = get_current_run_tree()
    langsmith_run_id = str(run_tree.id) if run_tree else None

    initial_state: SupportState = {
        "user_query": sample.question,
        "session_id": session_id,
        "intent": "",
        "retrieved_context": [],
        "draft_reply": "",
        "final_reply": "",
        "retry_count": 0,
        "metadata": sample.metadata,
        "qa_verdict": {},
        "requires_human_review": False,
        "human_feedback": None,
        "messages": [],
    }

    final_state: SupportState = await orchestrator.run(initial_state)

    return EvalRow(
        question=sample.question,
        answer=final_state.get("final_reply", ""),
        contexts=final_state.get("retrieved_context", []),
        ground_truth=sample.ground_truth,
        ground_truth_contexts=sample.ground_truth_contexts,
        intent=final_state.get("intent", ""),
        expected_intent=sample.expected_intent,
        retry_count=final_state.get("retry_count", 0),
        session_id=session_id,
        langsmith_run_id=langsmith_run_id,
    )


import asyncio

async def run_all_samples(
    orchestrator: OrchestratorAgent,
    samples: list[EvalSample],
    eval_run_id: str,
    max_concurrency: int = 5,
) -> list[EvalRow]:
    """Run the OrchestratorAgent concurrently over all eval samples.

    Failed samples are logged as warnings and recorded with ``intent="failed"``
    so they do not silently skew downstream metrics.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _process_sample(idx: int, sample: EvalSample) -> EvalRow:
        session_id = f"{eval_run_id}-sample-{idx:04d}"
        logger.info(
            "runner.run_sample",
            idx=idx,
            total=len(samples),
            question_preview=sample.question[:60],
        )

        async with semaphore:
            try:
                return await run_system_on_sample(orchestrator, sample, session_id)
            except Exception as exc:
                logger.warning(
                    "runner.sample_failed",
                    idx=idx,
                    session_id=session_id,
                    error=str(exc),
                )
                return EvalRow(
                    question=sample.question,
                    answer="",
                    contexts=[],
                    ground_truth=sample.ground_truth,
                    ground_truth_contexts=sample.ground_truth_contexts,
                    intent="failed",
                    expected_intent=sample.expected_intent,
                    retry_count=0,
                    session_id=session_id,
                    langsmith_run_id=None,
                )

    tasks = [_process_sample(idx, sample) for idx, sample in enumerate(samples)]
    rows = await asyncio.gather(*tasks)

    return list(rows)
