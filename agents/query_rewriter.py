# FILE: customer_support_rag/agents/query_rewriter.py
"""QueryRewriter — Contextual query rewriting for multi-turn conversations.

Responsibility:
    Given the N most recent conversation turns for a session and the current
    user query, detect whether the query is a follow-up that contains ambiguous
    references (pronouns, demonstratives, ellipsis) and, if so, rewrite it into
    a fully self-contained query before it reaches the retrieval pipeline.

Design principles:
    - ZERO impact on SupportState, orchestrator, or downstream agents.
    - Two-step process:
        1. Heuristic detection (free) — check for follow-up signal keywords.
           If none found, return the original query immediately (no LLM call).
        2. LLM rewriting (only when needed)
    - Returns a (rewritten_query, was_rewritten) tuple so callers can log
      when rewriting actually occurred.

Example:
    Turn 1: "My Dell laptop has a WiFi connectivity issue."
    Turn 2: "Does that affect my warranty?"
             ↓ rewriter
             "Does a WiFi connectivity issue on a Dell laptop affect my warranty?"
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from core.logging import get_logger

logger: structlog.BoundLogger = get_logger(__name__)

# ── Follow-up signal keywords ─────────────────────────────────────────────────
# If ANY of these appear in the query (case-insensitive), we attempt rewriting.
# Covers Vietnamese and English common reference patterns.
_FOLLOW_UP_SIGNALS: frozenset[str] = frozenset({
    # Vietnamese references
    "đó", "cái đó", "vấn đề đó", "điều đó", "nó", "cái này", "vấn đề này",
    "điều này", "như vậy", "như thế", "còn", "thêm về", "giải thích thêm",
    "nói thêm", "ý bạn", "ý trên",
    # English pronouns / demonstratives
    "that", "it", "this", "those", "these", "the same", "the issue",
    "the problem", "the error", "the case", "the matter",
    # English ellipsis phrases
    "explain more", "tell me more", "what about", "and also", "more detail",
    "what does that", "can you elaborate", "how about", "same thing",
})

# ── Prompt templates ──────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a query rewriting assistant for a customer support system.

Task: Given a conversation history and a follow-up query, rewrite the follow-up
query so it is fully self-contained and optimally formatted for a vector search engine.
You MUST resolve all pronouns, demonstratives, and implicit references.
CRITICALLY: You must enrich the query by injecting important contextual facts from the history (such as the user's name, age, product model, or specific constraints) ONLY IF they are essential to identifying the exact entities the user is asking about.

Rules:
- Output ONLY the rewritten query. No explanation, no prefix, no punctuation changes.
- If the query already explicitly contains all necessary context and facts, return it UNCHANGED (exact copy).
- Keep the rewritten query extremely concise. Do NOT add unnecessary background stories, previous intents (e.g. "who wants a refund"), or over-explain the user's situation.
- Preserve the original language (Vietnamese stays Vietnamese, English stays English).\
"""

_FINAL_HUMAN_TEMPLATE = """\
Based on the conversation history above, rewrite the following follow-up query so it is fully self-contained and optimally formatted for a vector search engine.

Follow-up query: {query}
Rewritten query:\
"""


class QueryRewriter:
    """Rewrites ambiguous follow-up queries using short conversation history.

    Attributes:
        _llm: A cheap, fast LLM (gpt-4o) used only when rewriting is needed.
    """

    def __init__(self) -> None:
        # Use gpt-4o-mini deliberately: rewriting is a simple task and cost matters.
        self._llm: ChatOpenAI = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,          # deterministic rewrites
            api_key=settings.openai_api_key,
            max_tokens=256,           # rewritten query should be short
        )

    async def rewrite(
        self,
        query: str,
        conversation_history: list[dict],
    ) -> tuple[str, bool]:
        """Rewrite a query if it appears to reference earlier conversation turns.

        Args:
            query: The raw user query for the current turn.
            conversation_history: Ordered list (oldest first) of previous turns, each a
                dict with keys ``"role"`` (user or assistant) and ``"content"``.

        Returns:
            A ``(rewritten_query, was_rewritten)`` tuple:
            - ``rewritten_query``: The enriched query (or the original if no
              rewriting was needed).
            - ``was_rewritten``: True if the LLM was actually invoked.
        """
        if not conversation_history:
            return query, False

        # ── Step 1: Heuristic gate (zero cost but can lead to low performance) ───────────────────────────────
        # lowered = query.lower()
        # has_signal = any(sig in lowered for sig in _FOLLOW_UP_SIGNALS)

        # if not has_signal:
        #     logger.debug("query_rewriter.skip", reason="no_follow_up_signal")
        #     return query, False

        # ── Step 2: LLM rewriting (always when signal detected) ────────────────
        logger.info(
            "query_rewriter.rewrite.start",
            query_preview=query[:80],
            num_turns=len(conversation_history) // 2,
        )
        start = time.monotonic()

        try:
            rewritten = await self._rewrite_with_retry(query, conversation_history)
        except Exception as exc:
            # On any failure, log and gracefully fall back to the original query.
            logger.warning(
                "query_rewriter.rewrite.failed",
                error=str(exc),
                fallback="original_query",
            )
            return query, False

        latency_ms = (time.monotonic() - start) * 1000
        logger.info(
            "query_rewriter.rewrite.done",
            original=query[:80],
            rewritten=rewritten[:80],
            latency_ms=round(latency_ms, 1),
        )
        return rewritten, True

    @retry(
        stop=stop_after_attempt(2),   # only 2 attempts — fail fast for latency
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    async def _rewrite_with_retry(
        self,
        query: str,
        conversation_history: list[dict],
    ) -> str:
        """Call the LLM to produce a self-contained rewrite of the query."""
        messages: list[BaseMessage] = [SystemMessage(content=_SYSTEM_PROMPT)]

        # Append native history messages
        for msg in conversation_history:
            content = msg.get("content", "")
            if msg.get("role") == "assistant":
                # Truncate long replies to keep the prompt lean for rewriting
                reply_preview = content[:200].rstrip()
                if len(content) > 200:
                    reply_preview += "..."
                messages.append(AIMessage(content=reply_preview))
            elif msg.get("role") == "user":
                messages.append(HumanMessage(content=content))

        # Append final request to rewrite the query
        final_human_content = _FINAL_HUMAN_TEMPLATE.format(query=query)
        messages.append(HumanMessage(content=final_human_content))

        response = await self._llm.ainvoke(messages)
        rewritten = response.content.strip()

        # Safety: if LLM returns empty string, fall back to original
        return rewritten if rewritten else query
