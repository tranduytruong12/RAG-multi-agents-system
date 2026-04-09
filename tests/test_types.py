"""Test suite for core/types.py — roundtrip serialization and contract validation.

Covers:
    - SupportState TypedDict field presence and type correctness
    - QAVerdict Pydantic model roundtrip (model_dump → re-instantiation)
    - QAVerdict boundary validation (confidence_score clamping)
    - AgentAction roundtrip
    - IngestionResult roundtrip
    - VALID_INTENTS frozenset membership and completeness
    - DocumentChunk and RetrievalResult dataclass construction
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.types import (
    VALID_INTENTS,
    AgentAction,
    DocumentChunk,
    IngestionResult,
    QAVerdict,
    RetrievalResult,
    SupportState,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_full_state(**overrides) -> SupportState:
    """Return a fully populated SupportState for testing."""
    base: SupportState = {
        "user_query": "How do I return a product?",
        "session_id": "sess-test-001",
        "intent": "refund",
        "retrieved_context": ["Context chunk 1.", "Context chunk 2."],
        "draft_reply": "Here is how to return a product.",
        "final_reply": "Here is how to return a product (approved).",
        "retry_count": 1,
        "metadata": {"trace_id": "trace-abc"},
        "requires_human_review": False,
        "human_feedback": None,
        "messages": [],
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SupportState
# ═══════════════════════════════════════════════════════════════════════════════

class TestSupportState:
    """Validate SupportState TypedDict structure."""

    def test_all_required_keys_present(self):
        """SupportState must contain all declared fields."""
        state = _make_full_state()
        required_keys = [
            "user_query", "session_id", "intent", "retrieved_context",
            "draft_reply", "final_reply", "retry_count", "metadata",
            "requires_human_review", "human_feedback", "messages",
        ]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"

    def test_retrieved_context_is_list_of_strings(self):
        """retrieved_context must be a list of str."""
        state = _make_full_state()
        assert isinstance(state["retrieved_context"], list)
        assert all(isinstance(c, str) for c in state["retrieved_context"])

    def test_human_feedback_allows_none(self):
        """human_feedback can be None."""
        state = _make_full_state(human_feedback=None)
        assert state["human_feedback"] is None

    def test_human_feedback_allows_string(self):
        """human_feedback can be a non-empty string."""
        state = _make_full_state(human_feedback="Please improve the tone.")
        assert state["human_feedback"] == "Please improve the tone."

    def test_messages_is_list(self):
        """messages field must be a list."""
        state = _make_full_state()
        assert isinstance(state["messages"], list)

    def test_metadata_is_dict(self):
        """metadata field must be a dict."""
        state = _make_full_state()
        assert isinstance(state["metadata"], dict)

    def test_retry_count_is_int(self):
        """retry_count must be an integer."""
        state = _make_full_state()
        assert isinstance(state["retry_count"], int)

    def test_requires_human_review_is_bool(self):
        """requires_human_review must be a bool."""
        state = _make_full_state()
        assert isinstance(state["requires_human_review"], bool)

    def test_roundtrip_dict_update(self):
        """SupportState can be updated with a partial dict (simulates LangGraph merge)."""
        state = _make_full_state()
        patch = {"intent": "billing", "retry_count": 2}
        state.update(patch)
        assert state["intent"] == "billing"
        assert state["retry_count"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 2. QAVerdict
# ═══════════════════════════════════════════════════════════════════════════════

class TestQAVerdict:
    """Validate QAVerdict Pydantic model serialization and constraints."""

    def test_passed_true_roundtrip(self):
        """A passing verdict serializes and deserializes correctly."""
        v = QAVerdict(passed=True, issues=[], confidence_score=0.98)
        d = v.model_dump()
        v2 = QAVerdict(**d)
        assert v2.passed is True
        assert v2.confidence_score == 0.98
        assert v2.issues == []

    def test_passed_false_with_issues(self):
        """A failing verdict includes all issue details."""
        v = QAVerdict(
            passed=False,
            issues=["Bad tone.", "Missing policy info."],
            bad_tone=True,
            missing_policy_info=True,
            confidence_score=0.6,
        )
        assert v.passed is False
        assert len(v.issues) == 2
        assert v.bad_tone is True
        assert v.missing_policy_info is True

    def test_inaccurate_flag(self):
        """inaccurate flag is correctly stored."""
        v = QAVerdict(passed=False, issues=["Invented facts."], inaccurate=True)
        assert v.inaccurate is True

    def test_confidence_score_lower_bound(self):
        """confidence_score of 0.0 is valid."""
        v = QAVerdict(passed=False, confidence_score=0.0)
        assert v.confidence_score == 0.0

    def test_confidence_score_upper_bound(self):
        """confidence_score of 1.0 is valid."""
        v = QAVerdict(passed=True, confidence_score=1.0)
        assert v.confidence_score == 1.0

    def test_confidence_score_out_of_range_raises(self):
        """confidence_score > 1.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            QAVerdict(passed=True, confidence_score=1.5)

    def test_confidence_score_negative_raises(self):
        """confidence_score < 0.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            QAVerdict(passed=False, confidence_score=-0.1)

    def test_default_flags_are_false(self):
        """All boolean flag fields default to False."""
        v = QAVerdict(passed=True)
        assert v.missing_policy_info is False
        assert v.bad_tone is False
        assert v.inaccurate is False

    def test_model_dump_to_dict_and_back(self):
        """model_dump() produces a plain dict that can reconstruct the model."""
        v = QAVerdict(
            passed=False,
            issues=["Issue A"],
            bad_tone=True,
            confidence_score=0.55,
        )
        d = v.model_dump()
        assert isinstance(d, dict)
        v2 = QAVerdict(**d)
        assert v2.issues == ["Issue A"]
        assert v2.bad_tone is True
        assert v2.confidence_score == pytest.approx(0.55)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. AgentAction
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentAction:
    """Validate AgentAction Pydantic model fields and serialization."""

    def test_roundtrip(self):
        """AgentAction serializes and deserializes correctly."""
        action = AgentAction(
            agent_name="DraftWriterAgent",
            action="draft_reply",
            input_summary="intent=refund, ctx_chunks=3",
            output_summary="draft length=120 chars",
            latency_ms=250.5,
            success=True,
        )
        d = action.model_dump()
        a2 = AgentAction(**d)
        assert a2.agent_name == "DraftWriterAgent"
        assert a2.success is True
        assert a2.latency_ms == pytest.approx(250.5)

    def test_success_false(self):
        """AgentAction with success=False is valid."""
        action = AgentAction(
            agent_name="QAAgent",
            action="verify_draft",
            input_summary="draft_reply=...",
            output_summary="passed=False",
            latency_ms=100.0,
            success=False,
        )
        assert action.success is False

    def test_model_dump_returns_dict(self):
        """model_dump() returns a plain Python dict."""
        action = AgentAction(
            agent_name="X", action="y", input_summary="in",
            output_summary="out", latency_ms=1.0, success=True,
        )
        d = action.model_dump()
        assert isinstance(d, dict)
        assert d["agent_name"] == "X"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. IngestionResult
# ═══════════════════════════════════════════════════════════════════════════════

class TestIngestionResult:
    """Validate IngestionResult Pydantic model."""

    def test_success_roundtrip(self):
        """A successful IngestionResult roundtrips correctly."""
        r = IngestionResult(
            total_documents=10,
            total_chunks=150,
            failed_documents=[],
            success=True,
            duration_seconds=12.5,
        )
        d = r.model_dump()
        r2 = IngestionResult(**d)
        assert r2.total_chunks == 150
        assert r2.success is True

    def test_failed_documents_list(self):
        """failed_documents contains paths of failed files."""
        r = IngestionResult(
            total_documents=5,
            total_chunks=40,
            failed_documents=["data/docs/broken.pdf"],
            success=False,
            duration_seconds=3.0,
        )
        assert len(r.failed_documents) == 1
        assert "broken.pdf" in r.failed_documents[0]

    def test_default_failed_documents_is_empty_list(self):
        """failed_documents defaults to empty list."""
        r = IngestionResult(
            total_documents=1, total_chunks=10, success=True, duration_seconds=1.0
        )
        assert r.failed_documents == []


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VALID_INTENTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidIntents:
    """Validate the VALID_INTENTS frozenset contract."""

    def test_is_frozenset(self):
        """VALID_INTENTS must be an immutable frozenset."""
        assert isinstance(VALID_INTENTS, frozenset)

    def test_contains_all_expected_intents(self):
        """All five canonical intents must be present."""
        expected = {"refund", "technical", "billing", "general", "escalate"}
        assert expected == VALID_INTENTS

    def test_refund_is_valid(self):
        assert "refund" in VALID_INTENTS

    def test_technical_is_valid(self):
        assert "technical" in VALID_INTENTS

    def test_unknown_intent_not_in_set(self):
        assert "unknown" not in VALID_INTENTS

    def test_immutable(self):
        """frozenset must reject mutation."""
        with pytest.raises(AttributeError):
            VALID_INTENTS.add("new_intent")  # type: ignore[attr-defined]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DocumentChunk and RetrievalResult
# ═══════════════════════════════════════════════════════════════════════════════

class TestDocumentChunk:
    """Validate DocumentChunk dataclass."""

    def test_construction(self):
        """DocumentChunk can be instantiated with required fields."""
        chunk = DocumentChunk(
            chunk_id="chunk-001",
            source_path="data/docs/faq.md",
            content="Our refund policy allows 30 days.",
        )
        assert chunk.chunk_id == "chunk-001"
        assert chunk.token_count == 0  # default

    def test_metadata_default_is_empty_dict(self):
        """metadata defaults to empty dict."""
        chunk = DocumentChunk(chunk_id="x", source_path="y", content="z")
        assert chunk.metadata == {}

    def test_token_count_set(self):
        """token_count can be set explicitly."""
        chunk = DocumentChunk(
            chunk_id="x", source_path="y", content="z", token_count=42
        )
        assert chunk.token_count == 42


class TestRetrievalResult:
    """Validate RetrievalResult dataclass."""

    def test_construction(self):
        """RetrievalResult requires chunk, score, and retrieval_method."""
        chunk = DocumentChunk(chunk_id="c1", source_path="f.md", content="text")
        result = RetrievalResult(chunk=chunk, score=0.87, retrieval_method="hybrid")
        assert result.score == pytest.approx(0.87)
        assert result.retrieval_method == "hybrid"

    def test_chunk_reference(self):
        """chunk attribute must hold the original DocumentChunk."""
        chunk = DocumentChunk(chunk_id="c2", source_path="g.md", content="more text")
        result = RetrievalResult(chunk=chunk, score=0.5, retrieval_method="dense")
        assert result.chunk is chunk
