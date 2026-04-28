# FILE: customer_support_rag/agents/qa_agent.py
"""QAAgent — Evaluates a draft reply against quality criteria.

Responsibility:
    Read `state["draft_reply"]` and `state["retrieved_context"]` from the shared
    SupportState. Use `ChatOpenAI.with_structured_output(QAVerdict)` to get a
    fully validated Pydantic QAVerdict back from the LLM.
    Return a partial SupportState dict so LangGraph can merge the QA result.

QA criteria (checked by the LLM):
    1. Tone — is the reply professional and empathetic?
    2. Accuracy — does the reply contradict or add facts not in the context?
    3. Policy completeness — does the reply include necessary policy details?
    4. Confidence — how confident is the model in its verdict (0-1 float)?
"""

from __future__ import annotations

import time

import structlog
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from core.exceptions import AgentError
from core.logging import get_logger
from core.types import AgentAction, QAVerdict, SupportState

logger: structlog.BoundLogger = get_logger(__name__)

# ── QA evaluation prompt ───────────────────────────────────────────────────────
# The LLM will evaluate the draft against these three dimensions.
# with_structured_output forces the model to fill all QAVerdict fields.

_SYSTEM_PROMPT = """You are a pragmatic Quality Assurance (QA) reviewer for customer-support replies.
Your job is to evaluate draft replies fairly, accounting for the fact that the knowledge base
may be incomplete — retrieval context may be sparse or absent for some queries.

EVALUATION WORKFLOW (process all three checks before deciding the final verdict):
1. TONE CHECK: Is the reply rude, robotic, dismissive, or completely lacking empathy?
   -> Only set `bad_tone` = true for genuinely problematic tone. Neutral or professional tone passes.
2. ACCURACY CHECK: Does the draft invent facts, hallucinate, or directly contradict the provided context?
   -> Only set `inaccurate` = true for clear contradictions or obvious hallucinations.
   -> If the context is sparse or absent and the draft responds honestly (e.g. "I'll escalate to a specialist"), do NOT flag as inaccurate.
3. POLICY CHECK: Does the draft omit a critical policy that is EXPLICITLY stated in the retrieved context?
   -> Only set `missing_policy_info` = true if the context clearly mentions a policy and the draft ignores it.
   -> If the context is sparse or missing, do NOT fail the draft on this check — it cannot include what is not in the KB.
4. ISSUE LOGGING: Populate `issues` only for genuine failures in the above checks.
   -> If a check passes, do not add it to `issues`.
5. FINAL VERDICT: Set `passed` = true if the draft passes at least 2 out of 3 checks AND is not inaccurate.
   -> A draft that is honest about knowledge gaps should generally pass.
   -> Reserve `passed` = false for drafts with clear hallucinations, rude tone, or missing an explicitly documented policy.

CRITICAL GUARDRAILS:
- CONTEXT-AWARE LENIENCY: When retrieved context is sparse or says "No relevant context found",
  weight TONE and ACCURACY checks only. Do not penalise for POLICY gaps the KB cannot fill.
- CONFIDENCE: Assign a `confidence_score` (0.0 to 1.0). Lower it if context is very sparse.
- HUMAN REVIEW: Only set `requires_human_review` = true for genuinely sensitive cases
  (legal threats, billing disputes, emotionally distressed customers) or confidence_score < 0.5.
"""

_HUMAN_TEMPLATE = """Context from knowledge base:
{context}

Draft reply to evaluate:
{draft_reply}

Evaluate and return your QAVerdict."""


class QAAgent:
    """Evaluates the draft reply and returns a structured QAVerdict.

    Fourth node in the LangGraph DAG. Uses structured output (function-calling)
    so the verdict is always a valid Pydantic model — no fragile string parsing.

    Attributes:
        _llm_with_verdict: ChatOpenAI instance bound to QAVerdict schema
            via with_structured_output(). Automatically enforces the response
            format using OpenAI function-calling.
        _prompt: ChatPromptTemplate with system + human message slots.
    """

    def __init__(self) -> None:
        _base_llm: ChatOpenAI = ChatOpenAI(
            model=settings.llm_model_name,
            temperature=0,
            api_key=settings.openai_api_key,
        )

        self._llm_with_verdict = _base_llm.with_structured_output(QAVerdict)

        self._prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_TEMPLATE),
        ])

    # ── Public API ─────────────────────────────────────────────────────────────

    async def verify(self, state: SupportState) -> dict:
        """LangGraph node entrypoint — evaluate the draft reply.

        Reads `draft_reply` and `retrieved_context` from state, calls the
        structured-output LLM, and returns a partial SupportState dict.

        The orchestrator uses the returned `qa_verdict` to decide whether to:
          - finalize the reply (passed=True)
          - retry drafting (passed=False, retry_count < max)
          - escalate to a human (passed=False, retry_count >= max)

        Args:
            state: Current shared LangGraph state. Must contain:
                - draft_reply (str)
                - retrieved_context (list[str])

        Returns:
            Partial SupportState dict with keys:
                - "qa_verdict": QAVerdict model serialised as dict (model_dump())
                - "messages": list with one AgentAction audit record

        Raises:
            AgentError: If the LLM call fails after all retries.
        """
        start = time.monotonic()
        draft_reply = state["draft_reply"]
        retrieved_context: list[str] = state["retrieved_context"]
        retry_count = state.get("retry_count", 0)
        logger.info("qa_agent.start", draft_length=len(draft_reply))

        context_str = "\n\n---\n\n".join(retrieved_context) if retrieved_context \
                        else "No relevant context found."

        try:
            verdict: QAVerdict = await self._verify_with_retry(context_str, draft_reply)
            
            # Suppress HITL until the final retry attempt so the drafter can
            # self-correct automatically first. Only the last retry may escalate
            # to a human reviewer. This restores the original smart suppression logic.
            if retry_count != settings.agent_max_retries - 1:
                logger.info("qa_agent.human_review_suppressed", retry_count=retry_count)
                verdict.requires_human_review = False

        except Exception as exc:
            logger.error("qa_agent.error", error=str(exc))
            raise AgentError("QA verification failed", context={"original_error": str(exc)})

        latency_ms = (time.monotonic() - start) * 1000

        # ── TESTING MODE: force human review regardless of LLM verdict ──────────
        # Remove this override when done testing HITL.
        # requires_human_review = True  # noqa: F841  ← TESTING — revert to: verdict.requires_human_review
        # ────────────────────────────────────────────────────────────────────────

        logger.info(
            "qa_agent.done",
            passed=verdict.passed,
            issues=verdict.issues,
            confidence=verdict.confidence_score,
            requires_human_review=verdict.requires_human_review,
            latency_ms=round(latency_ms, 1),
        )

        return {
            "qa_verdict": verdict.model_dump(),
            "requires_human_review": verdict.requires_human_review,
            "messages": [AgentAction(
                agent_name="QAAgent",
                action="verify_draft",
                input_summary=f"draft length={len(draft_reply)}",
                output_summary=(
                    f"passed={verdict.passed}, "
                    f"human_review={verdict.requires_human_review}, "
                    f"issues={len(verdict.issues)}"
                ),
                latency_ms=latency_ms,
                success=True,
            ).model_dump()]
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _verify_with_retry(self, context: str, draft_reply: str) -> QAVerdict:
        """Call the structured-output LLM and return a validated QAVerdict.

        Because `_llm_with_verdict` was created with `with_structured_output(QAVerdict)`,
        the chain automatically validates and coerces the LLM response into a
        QAVerdict Pydantic model. No manual JSON parsing needed.

        Args:
            context: Joined string of retrieved document chunks.
            draft_reply: The draft reply to be evaluated.

        Returns:
            A fully populated QAVerdict Pydantic model.

        Raises:
            Any exception after all tenacity retries are exhausted.
        """
        chain = self._prompt | self._llm_with_verdict
        response = await chain.ainvoke({"context": context, "draft_reply": draft_reply})
        return response
