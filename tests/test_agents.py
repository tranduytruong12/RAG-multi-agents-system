"""Test suite for all agents in the multi-agent RAG pipeline.

Covers:
    - IntentClassifierAgent: classify() happy path + LLM error → IntentClassificationError
    - DraftWriterAgent: draft() without feedback, with feedback, LLM error → AgentError
    - QAAgent: verify() passed verdict, failed verdict, LLM error → AgentError
    - OrchestratorAgent._route_after_qa: all four routing branches (finalize,
      draft_reply, escalate, human_review)

All LLM and retriever calls are fully mocked — no network calls are made.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.exceptions import AgentError, IntentClassificationError
from core.types import QAVerdict, SupportState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_state(**overrides) -> SupportState:
    """Return a minimal SupportState with sensible defaults."""
    base: SupportState = {
        "user_query": "I want a refund for my broken headphones.",
        "session_id": "test-session-001",
        "intent": "refund",
        "retrieved_context": [
            "Our refund policy allows returns within 30 days of purchase.",
            "Contact support@example.com to initiate a refund.",
        ],
        "draft_reply": "We're happy to help with your refund request.",
        "final_reply": "",
        "retry_count": 0,
        "metadata": {},
        "requires_human_review": False,
        "human_feedback": None,
        "messages": [],
    }
    base.update(overrides)
    return base


def _make_qa_verdict(**overrides) -> dict:
    """Return a serialised QAVerdict dict with all checks passing by default."""
    v = QAVerdict(
        passed=True,
        issues=[],
        missing_policy_info=False,
        bad_tone=False,
        inaccurate=False,
        confidence_score=0.95,
    )
    d = v.model_dump()
    d.update(overrides)
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# 1. IntentClassifierAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntentClassifierAgent:
    """Unit tests for IntentClassifierAgent.classify()."""

    @pytest.mark.asyncio
    async def test_classify_returns_valid_intent(self):
        """Happy path: LLM returns a valid IntentOutput → state updated correctly."""
        from agents.intent_classifier import IntentClassifierAgent, IntentOutput

        agent = IntentClassifierAgent.__new__(IntentClassifierAgent)

        # Mock the structured-output LLM chain
        mock_result = IntentOutput(intent="refund")
        agent._classify_with_retry = AsyncMock(return_value=mock_result)

        state = _make_state(user_query="I want my money back for order #123.")
        result = await agent.classify(state)

        assert result["intent"] == "refund"
        assert len(result["messages"]) == 1
        action = result["messages"][0]
        assert action["agent_name"] == "IntentClassifierAgent"
        assert action["action"] == "classify_intent"
        assert action["success"] is True

    @pytest.mark.asyncio
    async def test_classify_technical_intent(self):
        """classify() correctly propagates 'technical' intent from LLM."""
        from agents.intent_classifier import IntentClassifierAgent, IntentOutput

        agent = IntentClassifierAgent.__new__(IntentClassifierAgent)
        agent._classify_with_retry = AsyncMock(return_value=IntentOutput(intent="technical"))

        result = await agent.classify(_make_state(user_query="My app keeps crashing."))
        assert result["intent"] == "technical"

    @pytest.mark.asyncio
    async def test_classify_billing_intent(self):
        """classify() correctly propagates 'billing' intent."""
        from agents.intent_classifier import IntentClassifierAgent, IntentOutput

        agent = IntentClassifierAgent.__new__(IntentClassifierAgent)
        agent._classify_with_retry = AsyncMock(return_value=IntentOutput(intent="billing"))

        result = await agent.classify(_make_state(user_query="Why was I charged twice?"))
        assert result["intent"] == "billing"

    @pytest.mark.asyncio
    async def test_classify_escalate_intent(self):
        """classify() correctly propagates 'escalate' intent."""
        from agents.intent_classifier import IntentClassifierAgent, IntentOutput

        agent = IntentClassifierAgent.__new__(IntentClassifierAgent)
        agent._classify_with_retry = AsyncMock(return_value=IntentOutput(intent="escalate"))

        result = await agent.classify(_make_state(user_query="I want to talk to a manager NOW!"))
        assert result["intent"] == "escalate"

    @pytest.mark.asyncio
    async def test_classify_llm_error_raises_intent_classification_error(self):
        """If the LLM chain fails, classify() must raise IntentClassificationError."""
        from agents.intent_classifier import IntentClassifierAgent

        agent = IntentClassifierAgent.__new__(IntentClassifierAgent)
        agent._classify_with_retry = AsyncMock(side_effect=RuntimeError("OpenAI timeout"))

        with pytest.raises(IntentClassificationError):
            await agent.classify(_make_state())

    @pytest.mark.asyncio
    async def test_classify_output_summary_contains_intent(self):
        """The AgentAction output_summary must mention the classified intent."""
        from agents.intent_classifier import IntentClassifierAgent, IntentOutput

        agent = IntentClassifierAgent.__new__(IntentClassifierAgent)
        agent._classify_with_retry = AsyncMock(return_value=IntentOutput(intent="general"))

        result = await agent.classify(_make_state())
        assert "general" in result["messages"][0]["output_summary"]

    @pytest.mark.asyncio
    async def test_classify_input_summary_contains_query_length(self):
        """The AgentAction input_summary must include the query length."""
        from agents.intent_classifier import IntentClassifierAgent, IntentOutput

        agent = IntentClassifierAgent.__new__(IntentClassifierAgent)
        agent._classify_with_retry = AsyncMock(return_value=IntentOutput(intent="general"))

        query = "test query"
        result = await agent.classify(_make_state(user_query=query))
        assert str(len(query)) in result["messages"][0]["input_summary"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DraftWriterAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestDraftWriterAgent:
    """Unit tests for DraftWriterAgent.draft()."""

    @pytest.mark.asyncio
    async def test_draft_without_feedback(self):
        """No human_feedback → base_prompt is used, draft_reply is set."""
        from agents.drafter import DraftWriterAgent

        agent = DraftWriterAgent.__new__(DraftWriterAgent)
        agent._base_prompt = None
        agent._feedback_prompt = None
        agent._draft_with_retry = AsyncMock(return_value="Here is your refund info.")

        result = await agent.draft(_make_state(human_feedback=None))

        assert result["draft_reply"] == "Here is your refund info."
        assert len(result["messages"]) == 1
        action = result["messages"][0]
        assert action["agent_name"] == "DraftWriterAgent"
        assert action["action"] == "draft_reply"
        assert action["success"] is True

    @pytest.mark.asyncio
    async def test_draft_with_human_feedback(self):
        """human_feedback present → feedback_prompt is used."""
        from agents.drafter import DraftWriterAgent

        agent = DraftWriterAgent.__new__(DraftWriterAgent)
        agent._base_prompt = None
        agent._feedback_prompt = None
        agent._draft_with_retry = AsyncMock(return_value="Improved reply based on your feedback.")

        state = _make_state(human_feedback="Please be more concise.")
        result = await agent.draft(state)

        assert result["draft_reply"] == "Improved reply based on your feedback."
        # Verify _draft_with_retry was called with human_feedback kwarg
        call_kwargs = agent._draft_with_retry.call_args.kwargs
        assert call_kwargs.get("human_feedback") == "Please be more concise."

    @pytest.mark.asyncio
    async def test_draft_no_context_uses_fallback_string(self):
        """Empty retrieved_context → 'No relevant context found.' is passed to LLM."""
        from agents.drafter import DraftWriterAgent

        agent = DraftWriterAgent.__new__(DraftWriterAgent)
        agent._base_prompt = None
        agent._feedback_prompt = None
        captured_context = {}

        async def capture(**kwargs):
            captured_context["context"] = kwargs.get("context", "")
            return "fallback reply"

        agent._draft_with_retry = AsyncMock(side_effect=capture)
        await agent.draft(_make_state(retrieved_context=[]))

        assert captured_context["context"] == "No relevant context found."

    @pytest.mark.asyncio
    async def test_draft_multiple_context_chunks_joined(self):
        """Multiple context chunks must be joined with separator."""
        from agents.drafter import DraftWriterAgent

        agent = DraftWriterAgent.__new__(DraftWriterAgent)
        agent._base_prompt = None
        agent._feedback_prompt = None
        captured_context = {}

        async def capture(**kwargs):
            captured_context["context"] = kwargs.get("context", "")
            return "reply"

        agent._draft_with_retry = AsyncMock(side_effect=capture)
        await agent.draft(_make_state(retrieved_context=["Chunk A", "Chunk B"]))

        assert "Chunk A" in captured_context["context"]
        assert "Chunk B" in captured_context["context"]
        assert "---" in captured_context["context"]

    @pytest.mark.asyncio
    async def test_draft_llm_error_raises_agent_error(self):
        """LLM failure → AgentError is raised (not raw exception)."""
        from agents.drafter import DraftWriterAgent

        agent = DraftWriterAgent.__new__(DraftWriterAgent)
        agent._base_prompt = None
        agent._feedback_prompt = None
        agent._draft_with_retry = AsyncMock(side_effect=RuntimeError("connection refused"))

        with pytest.raises(AgentError):
            await agent.draft(_make_state())

    @pytest.mark.asyncio
    async def test_draft_output_summary_contains_length(self):
        """The AgentAction output_summary must mention draft character length."""
        from agents.drafter import DraftWriterAgent

        agent = DraftWriterAgent.__new__(DraftWriterAgent)
        agent._base_prompt = None
        agent._feedback_prompt = None
        reply = "A perfectly sized reply."
        agent._draft_with_retry = AsyncMock(return_value=reply)

        result = await agent.draft(_make_state())
        assert str(len(reply)) in result["messages"][0]["output_summary"]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. QAAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestQAAgent:
    """Unit tests for QAAgent.verify()."""

    @pytest.mark.asyncio
    async def test_verify_passed_verdict(self):
        """LLM returns a passing QAVerdict → result qa_verdict.passed is True."""
        from agents.qa_agent import QAAgent

        agent = QAAgent.__new__(QAAgent)
        verdict = QAVerdict(passed=True, issues=[], confidence_score=0.98)
        agent._verify_with_retry = AsyncMock(return_value=verdict)

        state = _make_state()
        result = await agent.verify(state)

        assert result["qa_verdict"]["passed"] is True
        assert result["qa_verdict"]["issues"] == []
        action = result["messages"][0]
        assert action["agent_name"] == "QAAgent"
        assert action["action"] == "verify_draft"
        assert action["success"] is True

    @pytest.mark.asyncio
    async def test_verify_failed_verdict_with_issues(self):
        """LLM returns a failing verdict with issues list."""
        from agents.qa_agent import QAAgent

        agent = QAAgent.__new__(QAAgent)
        verdict = QAVerdict(
            passed=False,
            issues=["Tone is too aggressive.", "Missing refund timeline."],
            bad_tone=True,
            missing_policy_info=True,
            confidence_score=0.7,
        )
        agent._verify_with_retry = AsyncMock(return_value=verdict)

        result = await agent.verify(_make_state())

        assert result["qa_verdict"]["passed"] is False
        assert len(result["qa_verdict"]["issues"]) == 2
        assert result["qa_verdict"]["bad_tone"] is True
        assert result["qa_verdict"]["missing_policy_info"] is True

    @pytest.mark.asyncio
    async def test_verify_inaccurate_verdict(self):
        """Inaccurate flag is correctly propagated in result."""
        from agents.qa_agent import QAAgent

        agent = QAAgent.__new__(QAAgent)
        verdict = QAVerdict(
            passed=False,
            issues=["Contains invented information."],
            inaccurate=True,
            confidence_score=0.6,
        )
        agent._verify_with_retry = AsyncMock(return_value=verdict)

        result = await agent.verify(_make_state())
        assert result["qa_verdict"]["inaccurate"] is True

    @pytest.mark.asyncio
    async def test_verify_context_joined_for_empty_list(self):
        """Empty retrieved_context → 'No relevant context found.' passed to LLM."""
        from agents.qa_agent import QAAgent

        agent = QAAgent.__new__(QAAgent)
        captured = {}

        async def capture(context, draft_reply):
            captured["context"] = context
            return QAVerdict(passed=True, confidence_score=1.0)

        agent._verify_with_retry = AsyncMock(side_effect=capture)
        await agent.verify(_make_state(retrieved_context=[]))

        assert captured["context"] == "No relevant context found."

    @pytest.mark.asyncio
    async def test_verify_llm_error_raises_agent_error(self):
        """LLM failure → AgentError is raised."""
        from agents.qa_agent import QAAgent

        agent = QAAgent.__new__(QAAgent)
        agent._verify_with_retry = AsyncMock(side_effect=RuntimeError("API unavailable"))

        with pytest.raises(AgentError):
            await agent.verify(_make_state())

    @pytest.mark.asyncio
    async def test_verify_output_summary_mentions_passed(self):
        """AgentAction output_summary must mention passed= value."""
        from agents.qa_agent import QAAgent

        agent = QAAgent.__new__(QAAgent)
        verdict = QAVerdict(passed=True, confidence_score=0.9)
        agent._verify_with_retry = AsyncMock(return_value=verdict)

        result = await agent.verify(_make_state())
        assert "passed=True" in result["messages"][0]["output_summary"]

    @pytest.mark.asyncio
    async def test_verify_confidence_score_bounds(self):
        """Confidence score at boundary values (0.0 and 1.0) is accepted."""
        from agents.qa_agent import QAAgent

        agent = QAAgent.__new__(QAAgent)

        for score in (0.0, 1.0):
            verdict = QAVerdict(passed=True, confidence_score=score)
            agent._verify_with_retry = AsyncMock(return_value=verdict)
            result = await agent.verify(_make_state())
            assert result["qa_verdict"]["confidence_score"] == score


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OrchestratorAgent – _route_after_qa (pure routing logic, no LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrchestratorRouting:
    """Unit tests for OrchestratorAgent._route_after_qa() routing logic.

    The routing function is a pure function of state — safe to call without
    building the full LangGraph graph or making any LLM calls.
    """

    @pytest.fixture
    def router(self):
        """Return the _route_after_qa method bound to a stub orchestrator."""
        from agents.orchestrator import OrchestratorAgent

        # Patch sub-agents so __init__ doesn't hit OpenAI / ChromaDB
        with (
            patch("agents.orchestrator.IntentClassifierAgent"),
            patch("agents.orchestrator.HybridRetriever"),
            patch("agents.orchestrator.DraftWriterAgent"),
            patch("agents.orchestrator.QAAgent"),
            patch("agents.orchestrator.OrchestratorAgent._build_graph"),
        ):
            orch = OrchestratorAgent()
        return orch._route_after_qa

    def test_route_finalize_when_qa_passed(self, router):
        """qa_verdict.passed=True → route to 'finalize'."""
        state = _make_state(
            qa_verdict=_make_qa_verdict(passed=True),
            retry_count=0,
            requires_human_review=False,
        )
        assert router(state) == "finalize"

    def test_route_draft_reply_on_first_retry(self, router):
        """QA failed, retry_count < max_retries → route to 'draft_reply'."""
        state = _make_state(
            qa_verdict=_make_qa_verdict(passed=False, issues=["Bad tone."]),
            retry_count=0,  # < default max_retries (typically 3)
            requires_human_review=False,
            human_feedback=None,
        )
        assert router(state) == "draft_reply"

    def test_route_escalate_after_max_retries(self, router):
        """QA failed, retry_count >= max_retries, no human_feedback → 'escalate'."""
        from config.settings import settings

        state = _make_state(
            qa_verdict=_make_qa_verdict(passed=False, issues=["Still wrong."]),
            retry_count=settings.agent_max_retries,  # at or beyond limit
            requires_human_review=False,
            human_feedback=None,
        )
        assert router(state) == "escalate"

    def test_route_human_review_when_flag_set(self, router):
        """requires_human_review=True → route to 'human_review' (HITL path)."""
        state = _make_state(
            qa_verdict=_make_qa_verdict(passed=False),
            retry_count=0,
            requires_human_review=True,
        )
        assert router(state) == "human_review"

    def test_route_draft_reply_on_human_feedback_regardless_of_count(self, router):
        """human_feedback present → always routes back to 'draft_reply', skips escalate."""
        from config.settings import settings

        state = _make_state(
            qa_verdict=_make_qa_verdict(passed=False),
            retry_count=settings.agent_max_retries + 99,  # would normally escalate
            requires_human_review=False,
            human_feedback="Please shorten the reply.",
        )
        assert router(state) == "draft_reply"

    def test_route_finalize_takes_priority_over_human_review_flag_cleared(self, router):
        """After HITL resume: requires_human_review=False, passed=True → 'finalize'."""
        state = _make_state(
            qa_verdict=_make_qa_verdict(passed=True),
            retry_count=1,
            requires_human_review=False,  # cleared after interrupt resume
            human_feedback="Looks good.",
        )
        # Even though human_feedback is present, passed=True wins → finalize
        assert router(state) == "finalize"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. OrchestratorAgent – node helpers (_node_finalize, _node_escalate)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrchestratorNodes:
    """Tests for stateless node helpers inside OrchestratorAgent."""

    @pytest.fixture
    def orchestrator(self):
        from agents.orchestrator import OrchestratorAgent

        with (
            patch("agents.orchestrator.IntentClassifierAgent"),
            patch("agents.orchestrator.HybridRetriever"),
            patch("agents.orchestrator.DraftWriterAgent"),
            patch("agents.orchestrator.QAAgent"),
            patch("agents.orchestrator.OrchestratorAgent._build_graph"),
        ):
            return OrchestratorAgent()

    @pytest.mark.asyncio
    async def test_node_finalize_sets_final_reply(self, orchestrator):
        """_node_finalize promotes draft_reply → final_reply."""
        state = _make_state(draft_reply="Approved draft text.", retry_count=2)
        result = await orchestrator._node_finalize(state)

        assert result["final_reply"] == "Approved draft text."
        action = result["messages"][0]
        assert action["action"] == "finalize"
        assert action["success"] is True

    @pytest.mark.asyncio
    async def test_node_escalate_sets_final_reply_with_session_id(self, orchestrator):
        """_node_escalate builds a graceful reply and includes session_id."""
        state = _make_state(session_id="session-xyz", retry_count=3)
        result = await orchestrator._node_escalate(state)

        assert "session-xyz" in result["final_reply"]
        action = result["messages"][0]
        assert action["action"] == "escalate"
        assert action["success"] is False

    @pytest.mark.asyncio
    async def test_node_finalize_audit_record_mentions_retry_count(self, orchestrator):
        """Finalize audit record mentions number of retries."""
        state = _make_state(draft_reply="Good reply.", retry_count=2)
        result = await orchestrator._node_finalize(state)
        assert "2" in result["messages"][0]["input_summary"]

    @pytest.mark.asyncio
    async def test_node_escalate_audit_record_mentions_retry_count(self, orchestrator):
        """Escalate audit record mentions number of retries."""
        state = _make_state(retry_count=5)
        result = await orchestrator._node_escalate(state)
        assert "5" in result["messages"][0]["input_summary"]