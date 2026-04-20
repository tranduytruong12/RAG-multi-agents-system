# FILE: customer_support_rag/agents/drafter.py
"""DraftWriterAgent — Generates a professional customer-support reply.

Responsibility:
    Read `state["retrieved_context"]`, `state["intent"]`, `state["user_query"]`,
    and optionally `state["human_feedback"]` from the shared SupportState.
    Call GPT-4o to draft a professional reply grounded in the retrieved context.
    Return a partial SupportState dict so LangGraph can merge `draft_reply`.
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
from core.types import AgentAction, SupportState

logger: structlog.BoundLogger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a professional and empathetic customer-support expert.
Your goal is to write a highly accurate, solution-focused reply to the customer.

Strict Guardrails (CRITICAL):
1. NO HALLUCINATION: You MUST ground your reply strictly in the provided "Context from knowledge base". 
2. UNKNOWN INFORMATION: If the provided context does NOT contain the answer, or says "No relevant context found", do NOT invent facts. Apologize honestly and state that you will forward their query to a human specialist.
3. NO CLICHÉS: Do NOT start with "Dear Customer", "I hope this email finds you well", or other generic corporate openers. Be conversational and direct.

Tone & Formatting Guidelines based on Intent:
- If intent is "technical": Use bullet points or numbered lists to explain steps clearly.
- If intent is "billing": Be precise with terms. Break down explanations logically.
- If intent is "escalate": Adopt a highly empathetic, apologetic, and urgent tone. Reassure them that their issue is being prioritized.
- If intent is "refund": Be clear about policies and timeframes mentioned in the context.

Format your response in plain text (or markdown for bullet points) suitable for a chat or email interface."""

_BASE_HUMAN_TEMPLATE = """Context from knowledge base:
{context}

Customer intent: {intent}
Customer query: {query}

Write a professional and friendly reply (ONLY base on the context):"""

_FEEDBACK_HUMAN_TEMPLATE = """Context from knowledge base:
{context}

Customer intent: {intent}
Customer query: {query}

A human reviewer rejected the previous draft with the following feedback:
  {human_feedback}

Taking the feedback into account, write an improved professional reply:"""


class DraftWriterAgent:
    """Generates a context-grounded draft reply for the customer query."""

    def __init__(self) -> None:
        self._llm: ChatOpenAI = ChatOpenAI(
            model=settings.llm_model_name,
            temperature=0.2, # drafter need some creativity to generate a good reply
            api_key=settings.openai_api_key,
        )

        self._base_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _BASE_HUMAN_TEMPLATE),
        ])

        self._feedback_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _FEEDBACK_HUMAN_TEMPLATE),
        ])

    async def draft(self, state: SupportState) -> dict:
        """LangGraph node entrypoint — generate a draft reply."""
        start = time.monotonic()
        query = state["user_query"]
        intent = state["intent"]
        retrieved_context: list[str] = state["retrieved_context"]
        human_feedback: str | None = state.get("human_feedback")

        logger.info(
            "drafter.start",
            intent=intent,
            context_chunks=len(retrieved_context),
            has_feedback=bool(human_feedback),
        )

        context_str = "\n\n---\n\n".join(retrieved_context) if retrieved_context else "No relevant context found."

        try:
            if human_feedback:
                draft = await self._draft_with_retry(
                    prompt=self._feedback_prompt,
                    context=context_str,
                    intent=intent,
                    query=query,
                    human_feedback=human_feedback,
                )
            else:
                draft = await self._draft_with_retry(
                    prompt=self._base_prompt,
                    context=context_str,
                    intent=intent,
                    query=query,
                )

        except Exception as exc:
            logger.error("drafter.error", error=str(exc))
            raise AgentError("Draft generation failed", context={"original_error": str(exc)})

        latency_ms = (time.monotonic() - start) * 1000
        logger.info("drafter.done", latency_ms=round(latency_ms, 1), draft_length=len(draft))

        return {
            "draft_reply": draft,
            "messages": [AgentAction(
                agent_name="DraftWriterAgent",
                action="draft_reply",
                input_summary=f"intent={intent}, ctx_chunks={len(retrieved_context)}",
                output_summary=f"draft length={len(draft)} chars",
                latency_ms=latency_ms,
                success=True,
            ).model_dump()]
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _draft_with_retry(
        self,
        prompt: ChatPromptTemplate,
        context: str,
        intent: str,
        query: str,
        human_feedback: str | None = None,
    ) -> str:
        """Call the LLM with the given prompt template and return the reply string."""
        chain = prompt | self._llm
        invoke_input = {"context": context, "intent": intent, "query": query}
        if human_feedback:
            invoke_input["human_feedback"] = human_feedback
        response = await chain.ainvoke(invoke_input)
        return response.content.strip()
