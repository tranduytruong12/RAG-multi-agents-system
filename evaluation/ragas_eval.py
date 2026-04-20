"""
FILE: customer_support_rag/evaluation/ragas_eval.py
====================================================
RAGAS-based retrieval and generation quality metrics.

Responsible for:
    - Converting EvalRows into the HuggingFace Dataset format expected by RAGAS
    - Running the five standard RAGAS metrics
    - Computing intent classification accuracy (custom metric)

Metrics computed:
    • context_precision     — are retrieved chunks relevant to the question?
    • context_recall        — do chunks cover the expected answer?
    • faithfulness          — is the answer grounded in retrieved context?
    • answer_relevancy      — is the answer semantically relevant to the question?
    • answer_correctness    — does the answer match the ground truth?
    • intent_accuracy       — % of correctly classified intents (custom)
"""

from __future__ import annotations

import ragas
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    Faithfulness,
    ResponseRelevancy,
    AnswerCorrectness,
)
from ragas import SingleTurnSample, EvaluationDataset
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from openai import OpenAI

from config.settings import settings
from core.types import EvalRow
from core.logging import get_logger

logger = get_logger(__name__)

# Metrics will be instantiated dynamically inside run_ragas_evaluation


def build_ragas_dataset(rows: list[EvalRow]) -> EvaluationDataset:
    """Convert EvalRow objects into a RAGAS 0.4+ EvaluationDataset."""
    samples = []
    
    for r in rows:
        samples.append(SingleTurnSample(
            user_input=r.question,
            response=r.answer if r.answer else "Failed to generate answer.",
            retrieved_contexts=r.contexts,
            reference_contexts=r.ground_truth_contexts,
            reference=r.ground_truth,
        ))

    logger.info("ragas_eval.dataset_built", num_rows=len(rows), ragas_version=ragas.__version__)
    return EvaluationDataset(samples)


def run_ragas_evaluation(
    dataset: EvaluationDataset,
) -> tuple[dict[str, float], list[dict]]:
    """Run RAGAS metrics over the evaluation dataset."""
    
    openai_client = OpenAI(api_key=settings.openai_api_key)
    
    ragas_llm = llm_factory("gpt-4o-mini", client=openai_client)
    ragas_embeddings = embedding_factory('openai', model=settings.embed_model_name, client=openai_client)

    # Patch for backwards-comptability with ragas.metrics (embed_query vs embed_text)
    if not hasattr(ragas_embeddings, "embed_query"):
        ragas_embeddings.embed_query = ragas_embeddings.embed_text
    if not hasattr(ragas_embeddings, "embed_documents"):
        ragas_embeddings.embed_documents = ragas_embeddings.embed_texts

    logger.info("ragas_eval.running", msg="Initializing metrics with LLM and Embeddings")

    metrics = [
        LLMContextPrecisionWithReference(llm=ragas_llm),
        LLMContextRecall(llm=ragas_llm),
        Faithfulness(llm=ragas_llm),
        ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
    ]

    result = ragas_evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
    )

    df = result.to_pandas()
    metric_names = [m.name for m in metrics]
    available_metrics = [m for m in metric_names if m in df.columns]

    if available_metrics:
        mean_scores    = df[available_metrics].mean().to_dict()
        detailed_scores = df[available_metrics].to_dict(orient="records")
    else:
        logger.warning("ragas_eval.no_metrics_found", columns=list(df.columns))
        mean_scores    = {}
        detailed_scores = [{} for _ in range(len(df))]

    logger.info("ragas_eval.done", mean_scores=mean_scores)
    return mean_scores, detailed_scores


def compute_intent_accuracy(rows: list[EvalRow]) -> float:
    """Compute intent classification accuracy over all labelled samples.

    Samples without an ``expected_intent`` are excluded from the calculation.

    Returns:
        A float in [0.0, 1.0], or 0.0 if no labelled samples exist.
    """
    correct = sum(
        1 for r in rows
        if r.expected_intent is not None and r.intent == r.expected_intent
    )
    labelled = sum(1 for r in rows if r.expected_intent is not None)

    accuracy = correct / labelled if labelled > 0 else 0.0
    logger.info("ragas_eval.intent_accuracy", correct=correct, labelled=labelled, accuracy=accuracy)
    return accuracy
