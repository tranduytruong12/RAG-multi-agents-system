# FILE: customer_support_rag/agents/intent_classifier.py
"""IntentClassifierAgent — Determines the intent of a customer query.

Responsibility:
    Read `state["user_query"]` and classify it into one of the valid intents.
    Returns a partial SupportState dict containing the `intent`.

Conversation history strategy:
    Prior turns are injected via LangChain's MessagesPlaceholder as actual
    HumanMessage / AIMessage objects rather than serialized text.  This preserves
    the ChatML role tokens (``<|im_start|>user`` / ``<|im_start|>assistant``) that
    GPT-4o uses to delimit turn boundaries — giving the model the same structural
    signal it was fine-tuned on, rather than relying on plain-text "Customer:" prefixes.
"""

from __future__ import annotations

import time
from typing import Literal

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from core.exceptions import IntentClassificationError
from core.logging import get_logger
from core.types import AgentAction, SupportState

logger: structlog.BoundLogger = get_logger(__name__)


def _build_history_messages(history: list[dict]) -> list[BaseMessage]:
    """Convert conversation history dicts into LangChain message objects.

    Returns an empty list when there is no history so the MessagesPlaceholder
    renders nothing and the prompt reduces to a simple single-turn call.

    Args:
        history: List of ``{"role": "user"|"assistant", "content": str}`` dicts
            sourced from ``state["conversation_history"]``.

    Returns:
        List of :class:`HumanMessage` / :class:`AIMessage` objects preserving
        native ChatML role tokens understood by GPT-4o.
    """
    messages: list[BaseMessage] = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


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

If prior conversation turns are present above the current query, use them to resolve
ambiguous or context-dependent follow-up queries (e.g. "What about that?",
"And for refunds?") before classifying.

Return the classification result."""

# MessagesPlaceholder inserts the prior turns as native role-delimited messages.
# When history is empty the placeholder renders nothing — prompt is single-turn.
_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="conversation_history", optional=True),
    ("human", "{query}"),
])


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
        self._prompt = _PROMPT

    async def classify(self, state: SupportState) -> dict:
        """LangGraph node entrypoint — determine query intent."""
        start = time.monotonic()
        query = state["user_query"]
        history_messages = _build_history_messages(state.get("conversation_history", []))

        logger.info(
            "intent_classifier.start",
            query_preview=query[:50],
            history_turns=len(history_messages),
        )

        try:
            result = await self._classify_with_retry(query, history_messages)
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
    async def _classify_with_retry(
        self,
        query: str,
        history_messages: list[BaseMessage],
    ) -> IntentOutput:
        """Executes the classification prompt and guarantees structured output.

        ``conversation_history`` is passed as a list of LangChain message objects.
        MessagesPlaceholder expands them into native role-delimited turns so GPT-4o
        sees the correct ChatML structure rather than serialized plain text.
        """
        chain = self._prompt | self._llm
        return await chain.ainvoke({
            "query": query,
            "conversation_history": history_messages,
        })
