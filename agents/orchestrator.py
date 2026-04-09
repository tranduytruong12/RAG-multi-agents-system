# FILE: customer_support_rag/agents/orchestrator.py
"""OrchestratorAgent — LangGraph StateGraph wiring all agents into a cyclic graph.

Responsibility:
    Build and compile the LangGraph StateGraph. Define nodes (each mapped to
    an agent method), define edges (including conditional routing logic), and
    expose a single `run(state)` coroutine as the public entry point.

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
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from config.settings import settings
from core.exceptions import HumanReviewRequired, QAFailureError
from core.logging import get_logger
from core.types import AgentAction, QAVerdict, SupportState
from retrieval.retriever import HybridRetriever

from .drafter import DraftWriterAgent
from .intent_classifier import IntentClassifierAgent
from .qa_agent import QAAgent

logger: structlog.BoundLogger = get_logger(__name__)


# ── Node return-type aliases (for conditional edge routing) ────────────────────
_RouteAfterQA = Literal["draft_reply", "finalize", "escalate", "human_review"]


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

        self._graph = self._build_graph()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(self, initial_state: SupportState) -> SupportState:
        """Invoke the compiled graph with an initial state and return the final state.

        Creates a LangGraph config dict with the session_id as the thread_id so
        MemorySaver can checkpoint conversation state per session.

        Args:
            initial_state: Fully populated SupportState dict (all required keys set).
                At minimum, `user_query` and `session_id` must be non-empty.

        Returns:
            The final SupportState after the graph reaches END (finalize node).
            All fields will be populated: intent, retrieved_context, draft_reply,
            final_reply, retry_count, messages, etc.

        Raises:
            HumanReviewRequired: If the graph pauses for HITL review.
            QAFailureError: If QA fails after max retries and escalation logic raises.
            AgentError: For any unrecoverable sub-agent failure.
        """
        config: dict = {"configurable": {"thread_id": initial_state["session_id"]}}

        logger.info(
            "orchestrator.run.start",
            session_id=initial_state["session_id"],
            query_preview=initial_state["user_query"][:80],
        )

        final_state: SupportState = await self._graph.ainvoke(initial_state, config=config)

        logger.info(
            "orchestrator.run.done",
            session_id=initial_state["session_id"],
            intent=final_state.get("intent"),
            retry_count=final_state.get("retry_count"),
        )

        return final_state

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
        #   "draft_reply"  → "draft_reply"    (retry: QA failed, retry_count < max)
        #   "escalate"     → "escalate"       (QA failed, retry_count >= max)
        #   "human_review" → END              (HITL: LangGraph interrupt handle)
        builder.add_conditional_edges(
            "qa_check",
            self._route_after_qa,
            {"finalize": "finalize", "draft_reply": "draft_reply",
             "escalate": "escalate", "human_review": END},
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
        """Node 2 — Run HybridRetriever and return retrieved context as plain strings.

        This is the ONLY node that may call LlamaIndex code (via HybridRetriever).
        All other nodes must NOT import LlamaIndex symbols.
        """
        start = time.monotonic()
        results = await self._retriever.retrieve(state["user_query"], top_k=5)
        latency_ms = (time.monotonic() - start) * 1000
        context_strings = [r.chunk.content for r in results]
        return {
            "retrieved_context": context_strings,
            "messages": [AgentAction(
                agent_name="HybridRetriever",
                action="retrieve",
                input_summary=f"query: {state['user_query'][:80]}",
                output_summary=f"retrieved {len(results)} chunks",
                latency_ms=latency_ms,
                success=True,
            ).model_dump()]
        }

    async def _node_draft_reply(self, state: SupportState) -> dict:
        """Node 3 — Run DraftWriterAgent and return the draft reply.

        Delegate to self._drafter.draft(state).
        Return the result directly.
        """
        return await self._drafter.draft(state)

    async def _node_qa_check(self, state: SupportState) -> dict:
        """Node 4 — Run QAAgent, handle HITL interrupt, increment retry_count.

        Reads `requires_human_review` from the QA result (not from the incoming
        state) because QAAgent is the one that sets this flag in its return dict.
        Using `state` here would read the stale pre-QA value.
        """
        result = await self._qa_agent.verify(state)
        verdict_dict = result.get("qa_verdict", {})
        result["retry_count"] = state["retry_count"] + 1

        # Read flag from the QA agent's output, not the stale incoming state.
        if result.get("requires_human_review"):
            # Pause execution and hand off to a human reviewer via LangGraph interrupt.
            # `interrupt()` checkpoints state and pauses the graph until resumed.
            human_feedback = interrupt({"reason": "human review required", "draft": state["draft_reply"]})
            result["human_feedback"] = human_feedback

            # Clear the flag so routing does not loop back into human_review again
            # after the graph is resumed with the human's input.
            result["requires_human_review"] = False

            updated_verdict = dict(verdict_dict)
            if human_feedback:
                # Human provided feedback → reject draft and request a new one
                updated_verdict["passed"] = False
                updated_verdict["issues"] = ["Human reviewer requested changes"]
            else:
                # Human approved with no comment → accept the draft
                updated_verdict["passed"] = True
                updated_verdict["issues"] = []
            result["qa_verdict"] = updated_verdict

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
        """Determine the next node after qa_check based on QA result and retry count.

        This function is passed to `builder.add_conditional_edges`. LangGraph
        calls it with the current state after qa_check executes, and routes to
        the node whose name matches the returned string.

        Routing logic:
            1. If requires_human_review → "human_review"  (HITL pause)
            2. If qa_verdict.passed     → "finalize"       (happy path)
            3. If retry_count < max     → "draft_reply"    (retry loop)
            4. Otherwise                → "escalate"       (max retries exceeded)

        Returns:
            One of: "finalize" | "draft_reply" | "escalate" | "human_review"
        """
        verdict_dict = state.get("qa_verdict", {})
        verdict = QAVerdict(**verdict_dict)
        
        # Nếu cờ này vẫn là True (chưa bị xóa ở qa_check), route thẳng ra END
        if state.get("requires_human_review"):
            return "human_review"
            
        if verdict.passed:
            return "finalize"
            
        # Ưu tiên quay về draft_reply nếu có feedback từ người dùng, bỏ qua retry_count
        if state.get("human_feedback"):
            return "draft_reply"
            
        if state.get("retry_count", 0) < settings.agent_max_retries:
            return "draft_reply"
            
        return "escalate"
