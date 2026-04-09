# FILE: customer_support_rag/agents/intent_classifier.py
"""IntentClassifierAgent — Determines the intent of a customer query.

Responsibility:
    Read `state["user_query"]` and classify it into one of the valid intents.
    Returns a partial SupportState dict containing the `intent`.
"""

from __future__ import annotations

import time
from typing import Literal

import structlog
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from core.exceptions import IntentClassificationError
from core.logging import get_logger
from core.types import AgentAction, SupportState

logger: structlog.BoundLogger = get_logger(__name__)

# Output schema enforcing valid choices via Pydantic
class IntentOutput(BaseModel):
    intent: Literal["refund", "technical", "billing", "general", "escalate"] = Field(
        description="The classified intent of the user's query."
    )

_SYSTEM_PROMPT = """You are an expert customer support routing assistant.
Analyze the user's query and classify it precisely into one of the following categories:
- refund: Requests for returning money or products.
- technical: Issues with the product, bugs, or usage help.
- billing: Questions about invoices, payments, subscriptions, or charges.
- escalate: Angry customers demanding a human manager or immediate human intervention.
- general: Everything else (e.g. shipping times, generic product queries).

Return the classification result."""

_HUMAN_TEMPLATE = """User Query: {query}"""


class IntentClassifierAgent:
    """Classifies user queries into specific predefined intents."""

    def __init__(self) -> None:
        _base_llm = ChatOpenAI(
            model=settings.llm_model_name,
            temperature=0,  # Zero for consistent, deterministic classification
            api_key=settings.openai_api_key,
        )
        
        # Force structured JSON output matching IntentOutput Pydantic model
        self._llm = _base_llm.with_structured_output(IntentOutput)
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_TEMPLATE),
        ])

    async def classify(self, state: SupportState) -> dict:
        """LangGraph node entrypoint — determine query intent."""
        start = time.monotonic()
        query = state["user_query"]
        
        logger.info("intent_classifier.start", query_preview=query[:50])
        
        try:
            result = await self._classify_with_retry(query)
            intent = result.intent
        except Exception as exc:
            logger.error("intent_classifier.error", error=str(exc))
            raise IntentClassificationError("Failed to classify intent") from exc
            
        latency_ms = (time.monotonic() - start) * 1000
        logger.info("intent_classifier.done", intent=intent, latency_ms=round(latency_ms, 1))
        
        return {
            "intent": intent,
            "messages": [AgentAction(
                agent_name="IntentClassifierAgent",
                action="classify_intent",
                input_summary=f"query_length={len(query)}",
                output_summary=f"intent={intent}",
                latency_ms=latency_ms,
                success=True,
            ).model_dump()]
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _classify_with_retry(self, query: str) -> IntentOutput:
        """Executes the classification prompt and guarantees structured output."""
        chain = self._prompt | self._llm
        return await chain.ainvoke({"query": query})
