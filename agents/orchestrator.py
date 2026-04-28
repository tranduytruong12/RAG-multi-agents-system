# FILE: customer_support_rag/agents/orchestrator.py
"""OrchestratorAgent — LangGraph StateGraph wiring all agents into a cyclic graph.

Responsibility:
    Build and compile the LangGraph StateGraph. Define nodes (each mapped to
    an agent method), define edges (including conditional routing logic), and
    expose `run(state)` and `resume(session_id, human_feedback)` as the public
    entry points.

Control flow (cyclic graph):
    classify_intent
         │
         ▼
      retrieve              ← calls HybridRetriever (LlamaIndex)
         │
         ▼
     draft_reply            ← calls DraftWriterAgent (with Retrieval Tool)
         │
         ▼
      qa_check              ← calls QAAgent
         │
    ┌────┴────────────────────────┐
    │  passed?                    │
    ▼ YES                     NO  ▼
 finalize          retry_count < max?
    │                   │YES            │NO
    │              draft_reply      escalate
    ▼
 END (returns final_reply)

Human-in-the-loop (HITL):
    When qa_check detects requires_human_review=True, it calls LangGraph's
    `interrupt()`, which checkpoints the state and pauses execution.
    A human can then POST to /chat/resume with human_feedback.
    LangGraph resumes from the checkpoint with the updated state.
"""

from __future__ import annotations

import time
from typing import Literal

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from config.settings import settings
from core.exceptions import HumanReviewRequired, QAFailureError
from core.logging import get_logger
from core.types import AgentAction, QAVerdict, SupportState
from retrieval.retriever import HybridRetriever

from .drafter import DraftWriterAgent
from .intent_classifier import IntentClassifierAgent
from .qa_agent import QAAgent
from .query_rewriter import QueryRewriter

logger: structlog.BoundLogger = get_logger(__name__)


# ── Node return-type aliases (for conditional edge routing) ────────────────────
_RouteAfterQA = Literal["draft_reply", "finalize", "escalate"]


class OrchestratorAgent:
    """Compiles and runs the LangGraph multi-agent with cyclic graph.

    Build the StateGraph once in __init__ (expensive: loads models, compiles
    graph). Then call run() for each incoming query — it simply invokes the
    already-compiled graph.

    Attributes:
        _intent_classifier: IntentClassifierAgent instance.
        _retriever: HybridRetriever instance (LlamaIndex).
        _drafter: DraftWriterAgent instance.
        _qa_agent: QAAgent instance.
        _graph: The compiled LangGraph CompiledStateGraph (with MemorySaver).
    """

    def __init__(self) -> None:

        self._intent_classifier: IntentClassifierAgent = IntentClassifierAgent()
        self._retriever: HybridRetriever = HybridRetriever()
        self._drafter: DraftWriterAgent = DraftWriterAgent()
        self._qa_agent: QAAgent = QAAgent()
        self._query_rewriter: QueryRewriter = QueryRewriter()

        self._graph = self._build_graph()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(self, initial_state: SupportState) -> SupportState:
        """Invoke the compiled graph with an initial state and return the final state.

        Creates a LangGraph config dict with the session_id as the thread_id so
        MemorySaver can checkpoint conversation state per session.

        `ainvoke()`returns the state dict with a special `__interrupt__` key
        containing a tuple of Interrupt objects. This method detects that key
        and re-raises GraphInterrupt so callers (chat.py) can handle HITL uniformly.

        Args:
            initial_state: Fully populated SupportState dict (all required keys set).
                At minimum, `user_query` and `session_id` must be non-empty.

        Returns:
            The final SupportState after the graph reaches END (finalize node).
            All fields will be populated: intent, retrieved_context, draft_reply,
            final_reply, retry_count, messages, etc.

        Raises:
            GraphInterrupt: If the graph pauses at an interrupt() call (HITL).
                Callers must catch this and redirect the user to POST /chat/resume.
            QAFailureError: If QA fails after max retries and escalation logic raises.
            AgentError: For any unrecoverable sub-agent failure.
        """
        config: dict = {"configurable": {"thread_id": initial_state["session_id"]}}

        logger.info(
            "orchestrator.run.start",
            session_id=initial_state["session_id"],
            query_preview=initial_state["user_query"][:80],
        )

        result: dict = await self._graph.ainvoke(initial_state, config=config)

        # interrupt() does NOT raise — it embeds __interrupt__
        # in the return dict. Detect it and re-raise so chat.py catches it.
        self._raise_if_interrupted(result, initial_state["session_id"])

        final_state: SupportState = result  # type: ignore[assignment]

        logger.info(
            "orchestrator.run.done",
            session_id=initial_state["session_id"],
            intent=final_state.get("intent"),
            retry_count=final_state.get("retry_count"),
        )

        return final_state

    async def resume(self, session_id: str, human_feedback: str) -> SupportState:
        """Resume a graph that was paused by a HITL interrupt().

        LangGraph stores the checkpoint under `session_id` (as thread_id) in
        MemorySaver. This method re-invokes the graph with Command(resume=...),
        which injects `human_feedback` as the return value of the `interrupt()`
        call that originally paused the graph.

        The node re-executes from the top (LangGraph v1.x behaviour). The second
        call to interrupt() inside _node_qa_check will return `human_feedback`
        immediately instead of pausing again.

        Args:
            session_id: Must match the session_id used in the original run() call
                so MemorySaver locates the correct checkpoint.
            human_feedback: Free-text reviewer comment. Pass an empty string to
                approve the current draft without changes.

        Returns:
            The final SupportState after the resumed graph reaches END.

        Raises:
            GraphInterrupt: If a second interrupt() is hit (e.g. another HITL cycle
                was triggered). Callers should handle this the same way as in run().
            KeyError: If `session_id` has no checkpoint (e.g. session expired or
                never started).
        """
        config: dict = {"configurable": {"thread_id": session_id}}

        logger.info(
            "orchestrator.resume.start",
            session_id=session_id,
            has_feedback=bool(human_feedback),
        )

        result: dict = await self._graph.ainvoke(
            Command(resume=human_feedback),
            config=config,
        )

        final_state: SupportState = result  # type: ignore[assignment]

        logger.info(
            "orchestrator.resume.done",
            session_id=session_id,
            intent=final_state.get("intent"),
            retry_count=final_state.get("retry_count"),
        )

        return final_state

    # ── Private helpers ────────────────────────────────────────────────────────

    def _raise_if_interrupted(self, result: dict, session_id: str) -> None:
        """Re-raise GraphInterrupt if LangGraph v1.x embedded one in the result.

        In LangGraph >= 1.0, `ainvoke()` no longer raises GraphInterrupt directly.
        Instead, when a node calls interrupt(), the graph returns early with a
        synthetic `__interrupt__` key in the returned dict. This helper detects
        that key and re-raises GraphInterrupt so our callers (chat.py) can catch
        it in a single uniform `except GraphInterrupt` block.

        Args:
            result: The raw dict returned by ainvoke().
            session_id: Used for structured log context only.

        Raises:
            GraphInterrupt: If `__interrupt__` is present in `result`.
        """
        interrupts = result.get("__interrupt__")
        if not interrupts:
            return

        logger.info(
            "orchestrator.hitl.interrupt_detected",
            session_id=session_id,
            interrupt_count=len(interrupts),
        )
        # Re-raise with the same args structure chat.py already expects:
        # exc.args[0] is a tuple/list of Interrupt objects with a .value attribute.
        raise GraphInterrupt(tuple(interrupts))

    # ── Graph construction ─────────────────────────────────────────────────────

    def _build_graph(self):
        """Build, wire, and compile the LangGraph StateGraph.

        Returns:
            A compiled LangGraph runnable (CompiledStateGraph) with MemorySaver
            checkpointing enabled.
        """
        builder = StateGraph(SupportState)

        builder.add_node("classify_intent", self._node_classify_intent)
        builder.add_node("retrieve",        self._node_retrieve)
        builder.add_node("draft_reply",     self._node_draft_reply)
        builder.add_node("qa_check",        self._node_qa_check)
        builder.add_node("finalize",        self._node_finalize)
        builder.add_node("escalate",        self._node_escalate)

        builder.add_edge(START, "classify_intent")
        builder.add_edge("classify_intent", "retrieve")
        builder.add_edge("retrieve", "draft_reply")
        builder.add_edge("draft_reply", "qa_check")
        builder.add_edge("finalize", END)
        builder.add_edge("escalate", END)

        # Mapping:
        #   "finalize"     → "finalize"       (QA passed)
        #   "draft_reply"  → "draft_reply"    (QA failed, retry_count < max)
        #   "escalate"     → "escalate"       (QA failed, retry_count >= max)
        builder.add_conditional_edges(
            "qa_check",
            self._route_after_qa,
            {"finalize": "finalize", "draft_reply": "draft_reply",
             "escalate": "escalate"},
        )

        checkpointer = MemorySaver()
        return builder.compile(checkpointer=checkpointer)

    # ── Node functions ─────────────────────────────────────────────────────────

    async def _node_classify_intent(self, state: SupportState) -> dict:
        """Node 1 — Run IntentClassifierAgent and return partial state.

        Return the result directly (it's already a partial dict).
        """
        return await self._intent_classifier.classify(state)

    async def _node_retrieve(self, state: SupportState) -> dict:
        """Node 2 — Run QueryRewriter and HybridRetriever and return retrieved context.

        This node first rewrites the query using conversation history (if enabled),
        then calls LlamaIndex HybridRetriever to fetch context.
        All other nodes must NOT import LlamaIndex symbols.
        """
        # Lazy initialization: build the fusion index only on the first call.
        if self._retriever._fusion_retriever is None:
            logger.info("orchestrator.retriever.build_index.start")
            await self._retriever.build_index()
            logger.info("orchestrator.retriever.build_index.done")

        messages = []
        query_to_retrieve = state["user_query"]

        if settings.enable_query_rewriting and state.get("conversation_history"):
            start_rewrite = time.monotonic()
            query_to_retrieve, was_rewritten = await self._query_rewriter.rewrite(
                state["user_query"], state["conversation_history"]
            )
            if was_rewritten:
                latency_rewrite = (time.monotonic() - start_rewrite) * 1000
                messages.append(AgentAction(
                    agent_name="QueryRewriter",
                    action="rewrite",
                    input_summary=f"original: {state['user_query'][:80]}",
                    output_summary=f"rewritten: {query_to_retrieve[:80]}",
                    latency_ms=latency_rewrite,
                    success=True,
                ).model_dump())

        start = time.monotonic()
        results = await self._retriever.retrieve(query_to_retrieve, top_k=5)
        latency_ms = (time.monotonic() - start) * 1000
        context_strings = [r.chunk.content for r in results]
        
        messages.append(AgentAction(
            agent_name="HybridRetriever",
            action="retrieve",
            input_summary=f"query: {query_to_retrieve[:80]}",
            output_summary=f"retrieved {len(results)} chunks",
            latency_ms=latency_ms,
            success=True,
        ).model_dump())

        return {
            "retrieved_context": context_strings,
            "messages": messages
        }

    async def _node_draft_reply(self, state: SupportState) -> dict:
        """Node 3 — Run DraftWriterAgent and return the draft reply.

        Delegate to self._drafter.draft(state).
        Return the result directly.
        """
        return await self._drafter.draft(state)

    async def _node_qa_check(self, state: SupportState) -> dict:
        """Node 4 — Run QAAgent, handle HITL interrupt, increment retry_count.

        LangGraph v1.x resume behaviour: when the graph is resumed with
        Command(resume=...) after an interrupt(), this entire node is re-executed
        from the top. The second call to interrupt() will return the human's value
        immediately (no pause). To avoid a redundant LLM call on resume, we check
        whether the state already has a human_feedback value set by a prior cycle
        before calling the QA LLM again.

        Reads `requires_human_review` from the QA result (not from the incoming
        state) because QAAgent is the one that sets this flag in its return dict.
        """
        result = await self._qa_agent.verify(state)
        verdict_dict = result.get("qa_verdict", {})
        result["retry_count"] = state["retry_count"] + 1

        # Read flag from the QA agent's fresh output, not the stale incoming state.
        if result.get("requires_human_review"):
            # interrupt() on first call: raises GraphInterrupt (embedded in __interrupt__
            # return key in LangGraph v1.x). On resume, returns human_feedback directly.
            human_feedback = interrupt({"reason": "human review required", "draft": state["draft_reply"]})
            result["human_feedback"] = human_feedback

            # Clear the flag so routing does not trigger another human review loop
            # on the next QA cycle after the draft is re-generated.
            result["requires_human_review"] = False

            updated_verdict = dict(verdict_dict)
            if human_feedback:
                # Human provided feedback → reject draft, request a new one
                updated_verdict["passed"] = False
                updated_verdict["issues"] = ["Human reviewer requested changes: " + human_feedback]
            else:
                # Human approved (empty string) → accept draft as-is
                updated_verdict["passed"] = True
                updated_verdict["issues"] = []
            result["qa_verdict"] = updated_verdict

            logger.info(
                "orchestrator.hitl.feedback_received",
                session_id=state["session_id"],
                has_feedback=bool(human_feedback),
                verdict_passed=updated_verdict["passed"],
            )

        return result

    async def _node_finalize(self, state: SupportState) -> dict:
        """Node 5 — Promote draft_reply → final_reply and close the session.

        """
        return {
            "final_reply": state["draft_reply"],
            "messages": [AgentAction(
                agent_name="OrchestratorAgent",
                action="finalize",
                input_summary=f"draft approved after {state['retry_count']} retries",
                output_summary=f"final_reply set, length={len(state['draft_reply'])}",
                latency_ms=0.0,
                success=True,
            ).model_dump()]
        }

    async def _node_escalate(self, state: SupportState) -> dict:
        """Node 6 — Generate a graceful escalation reply and close session.

        Called when QA fails after max_retries. Sets a safe fallback reply
        that informs the customer a human agent will follow up.

        NOTE: Do NOT set `requires_human_review=True` here. That flag is
        semantically reserved for the HITL interrupt path triggered by QAAgent.
        Escalation is a separate terminal path that ends at END without pausing.
        """
        escalation_reply = (
            f"We apologise for the inconvenience. A human agent will review your "
            f"request and follow up shortly. Reference: {state['session_id']}"
        )
        logger.warning(
            "orchestrator.escalate",
            session_id=state["session_id"],
            retry_count=state["retry_count"],
        )
        return {
            "final_reply": escalation_reply,
            "messages": [AgentAction(
                agent_name="OrchestratorAgent",
                action="escalate",
                input_summary=f"QA failed after {state['retry_count']} retries",
                output_summary="escalation reply set",
                latency_ms=0.0,
                success=False,
            ).model_dump()]
        }

    # ── Conditional routing ────────────────────────────────────────────────────

    def _route_after_qa(self, state: SupportState) -> _RouteAfterQA:
        """Determine the next node after qa_check based on QA result and retry count."""
        verdict_dict = state.get("qa_verdict", {})

        # Safety guard: if qa_verdict is empty or malformed, treat as failed.
        if not verdict_dict or "passed" not in verdict_dict:
            logger.warning("orchestrator.route_after_qa.empty_verdict — defaulting to retry/escalate")
            verdict = QAVerdict(passed=False)
        else:
            verdict = QAVerdict(**verdict_dict)

        if verdict.passed:
            return "finalize"

        if state.get("human_feedback"):
            return "draft_reply"

        if state.get("retry_count", 0) < settings.agent_max_retries:
            return "draft_reply"

        return "escalate"
