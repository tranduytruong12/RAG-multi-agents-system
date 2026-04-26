# FILE: customer_support_rag/agents/drafter.py
"""DraftWriterAgent — Generates a professional customer-support reply.

Responsibility:
    Read `state["retrieved_context"]`, `state["intent"]`, `state["user_query"]`,
    and optionally `state["human_feedback"]` from the shared SupportState.
    Call GPT-4o to draft a professional reply grounded in the retrieved context.
    Return a partial SupportState dict so LangGraph can merge `draft_reply`.

Conversation history strategy:
    Prior turns are injected as actual HumanMessage / AIMessage objects via
    LangChain's MessagesPlaceholder.  This preserves the ChatML role tokens that
    GPT-4o was fine-tuned on, giving the model proper structural signal for
    turn-boundary detection rather than relying on plain-text "Customer:" prefixes.

    The final human message carries the RAG context + intent + current query so
    retrieval-grounded information is clearly scoped to the current turn only.
"""

from __future__ import annotations

import time

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from core.exceptions import AgentError
from core.logging import get_logger
from core.types import AgentAction, SupportState

logger: structlog.BoundLogger = get_logger(__name__)


def _build_history_messages(history: list[dict]) -> list[BaseMessage]:
    """Convert conversation history dicts into LangChain message objects.

    Returns an empty list when there is no history so MessagesPlaceholder
    renders nothing — the prompt degrades cleanly to a single-turn call.

    Args:
        history: List of ``{"role": "user"|"assistant", "content": str}`` dicts
            sourced from ``state["conversation_history"]``.

    Returns:
        List of :class:`HumanMessage` / :class:`AIMessage` preserving native
        ChatML role tokens understood by GPT-4o.
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

Format your response in plain text (or markdown for bullet points) suitable for a chat or email interface.
If prior conversation turns are present, maintain natural continuity — avoid re-explaining things
already covered and acknowledge prior context where relevant."""

# ── Prompt templates ───────────────────────────────────────────────────────────
# MessagesPlaceholder is used for the dynamic history section.  When history is
# empty the placeholder is omitted entirely — the prompt degrades to single-turn.
# The final HumanMessage carries RAG context + intent + current query, keeping
# retrieval-grounded information clearly scoped to the current request only.

_BASE_FINAL_HUMAN = """Context from knowledge base:
{context}

Customer intent: {intent}
Customer query: {query}

Write a professional and friendly reply (ONLY based on the context):"""

_FEEDBACK_FINAL_HUMAN = """Context from knowledge base:
{context}

Customer intent: {intent}
Customer query: {query}

A human reviewer rejected the previous draft with the following feedback:
  {human_feedback}

Taking the feedback into account (MUST follow human feedback!), write an improved professional reply:"""


class DraftWriterAgent:
    """Generates a context-grounded draft reply for the customer query."""

    def __init__(self) -> None:
        self._llm: ChatOpenAI = ChatOpenAI(
            model=settings.llm_model_name,
            temperature=0.2,  # drafter needs some creativity to generate a good reply
            api_key=settings.openai_api_key,
        )
        # System message is static — built once and reused across all requests.
        self._system_msg = SystemMessage(content=_SYSTEM_PROMPT)

    async def draft(self, state: SupportState) -> dict:
        """LangGraph node entrypoint — generate a draft reply."""
        start = time.monotonic()
        query = state["user_query"]
        intent = state["intent"]
        retrieved_context: list[str] = state["retrieved_context"]
        human_feedback: str | None = state.get("human_feedback")
        history_messages = _build_history_messages(state.get("conversation_history", []))

        logger.info(
            "drafter.start",
            intent=intent,
            context_chunks=len(retrieved_context),
            has_feedback=bool(human_feedback),
            history_turns=len(history_messages),
        )

        context_str = "\n\n---\n\n".join(retrieved_context) if retrieved_context else "No relevant context found."

        try:
            if human_feedback:
                draft = await self._draft_with_retry(
                    context=context_str,
                    intent=intent,
                    query=query,
                    history_messages=history_messages,
                    human_feedback=human_feedback,
                )
            else:
                draft = await self._draft_with_retry(
                    context=context_str,
                    intent=intent,
                    query=query,
                    history_messages=history_messages,
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
        context: str,
        intent: str,
        query: str,
        history_messages: list[BaseMessage],
        human_feedback: str | None = None,
    ) -> str:
        """Build the message list dynamically and invoke the LLM.

        Message structure sent to GPT-4o:
            [SystemMessage]
            [HumanMessage(turn_1), AIMessage(turn_1), ...]  ← history (may be empty)
            [HumanMessage(context + intent + current_query)] ← always last

        The history turns use native ChatML role tokens; the final human message
        keeps all RAG context clearly scoped to the current request.

        Args:
            context: Joined retrieved document chunks.
            intent: Classified intent label.
            query: Current customer query.
            history_messages: Prior turns as LangChain message objects.
            human_feedback: Optional reviewer feedback for the retry cycle.
        """
        final_human_template = _FEEDBACK_FINAL_HUMAN if human_feedback else _BASE_FINAL_HUMAN
        final_human_content = final_human_template.format(
            context=context,
            intent=intent,
            query=query,
            **({"human_feedback": human_feedback} if human_feedback else {}),
        )

        messages: list[BaseMessage] = [
            self._system_msg,
            *history_messages,                    # ← native role-delimited history
            HumanMessage(content=final_human_content),  # ← RAG context + current query
        ]

        response = await self._llm.ainvoke(messages)
        return response.content.strip()
