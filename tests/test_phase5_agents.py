"""Unit and integration tests for Phase 5 Multi-Agent System Architecture.

Tests:
1. Pydantic state schema, hand-offs, and ReasonCode enumerations
2. Script and Language Agent detection (Devanagari vs Hinglish vs English)
3. Query Agent transliteration and retry reformulation
4. Supervisor budget enforcement (max 8 LLM calls, max 2 retrieval rounds)
5. ScoreGate (System S1) non-agentic filtering
6. EvidenceGrader (System S3-S5) distractor filtering and abstention
7. VerifierAgent grounding, citation validation, and hallucination detection
8. AbstentionArbiter bounded retries and termination conditions
9. Complete System Ladder execution (S0 through S5) via LangGraph StateGraph
10. Offline Judge Agent strict isolation from serving graph
11. Zero evaluation labels or gold answer leakage in serving state
"""

from __future__ import annotations

import pytest

from indicrag.agents.abstention_arbiter import AbstentionArbiter
from indicrag.agents.evidence_grader import EvidenceGraderAgent
from indicrag.agents.graph import build_indicrag_graph
from indicrag.agents.query_agent import QueryAgent
from indicrag.agents.score_gate import ScoreGate
from indicrag.agents.script_detector import ScriptLanguageAgent
from indicrag.agents.state import (
    IndicRAGState,
    LanguageType,
    ReasonCode,
    ScriptType,
    SystemLadderConfig,
    VerificationVerdict,
)
from indicrag.agents.stub_client import StubAnswererGenerator, create_stub_retriever
from indicrag.agents.supervisor import SupervisorAgent
from indicrag.eval.judge import JudgeAgent


def test_pydantic_state_schema_and_enums():
    """Verify IndicRAGState structure, default values, and ReasonCode specifications."""
    state = IndicRAGState(
        query_id="dev_001",
        question="भारत की राजधानी क्या है?",
    )
    assert state.query_id == "dev_001"
    assert state.llm_calls == 0
    assert state.retrieval_rounds == 0
    assert state.max_llm_calls == 8
    assert state.max_retrieval_rounds == 2
    assert state.retry_count == 0
    assert state.is_abstention is False
    assert state.reason_code is None

    # Verify all required ReasonCodes exist per AGENTS.md and Master Brief
    expected_codes = {
        "NO_EVIDENCE",
        "UNSUPPORTED_ANSWER",
        "ANSWERER_REFUSED",
        "LOW_CONFIDENCE",
        "BUDGET_EXHAUSTED",
        "CONTRADICTORY_EVIDENCE",
    }
    actual_codes = {code.value for code in ReasonCode}
    assert expected_codes.issubset(actual_codes)


def test_script_language_agent_detection():
    """Verify script and language detection across Devanagari, Hinglish, and English."""
    agent = ScriptLanguageAgent()

    # 1. Pure Devanagari
    state_hi = IndicRAGState(query_id="q1", question="भारत का राष्ट्रीय पक्षी कौन सा है?")
    state_hi = agent.run(state_hi)
    assert state_hi.script == ScriptType.DEVANAGARI
    assert state_hi.language == LanguageType.HINDI

    # 2. Roman Hinglish with lexical Hindi markers
    state_hinglish = IndicRAGState(query_id="q2", question="bharat ki rajdhani kya hai?")
    state_hinglish = agent.run(state_hinglish)
    assert state_hinglish.script == ScriptType.ROMAN_HINGLISH
    assert state_hinglish.language == LanguageType.HINGLISH

    # 3. Pure Latin English
    state_en = IndicRAGState(query_id="q3", question="What is the capital of India?")
    state_en = agent.run(state_en)
    assert state_en.script == ScriptType.LATIN_ENGLISH
    assert state_en.language == LanguageType.ENGLISH


def test_query_agent_transliteration_and_reformulation():
    """Verify query agent performs rule-based transliteration and retry reformulation."""
    agent = QueryAgent()

    # Transliteration test
    state = IndicRAGState(
        query_id="q1",
        question="bharat ki rajdhani kya hai",
        script=ScriptType.ROMAN_HINGLISH,
    )
    state = agent.run(state)
    assert "भारत" in state.active_query
    assert "राजधानी" in state.active_query
    assert len(state.reformulated_queries) >= 1

    # Retry reformulation test
    state._needs_retry = True
    state.reason_code = ReasonCode.UNSUPPORTED_ANSWER
    state = agent.run(state)
    assert len(state.reformulated_queries) >= 2


def test_supervisor_budget_enforcement():
    """Verify Supervisor stops execution when LLM call or retrieval budgets are exceeded."""
    supervisor = SupervisorAgent(max_llm_calls=8, max_retrieval_rounds=2)

    # Within budget
    ok_state = IndicRAGState(query_id="q1", question="test", llm_calls=4, retrieval_rounds=1)
    is_ex, msg = supervisor.check_budget(ok_state)
    assert is_ex is False

    # LLM call budget hit (8 calls)
    ex_state = IndicRAGState(query_id="q2", question="test", llm_calls=8, retrieval_rounds=1)
    is_ex, msg = supervisor.check_budget(ex_state)
    assert is_ex is True
    assert "LLM call budget exhausted" in msg

    handled_state = supervisor.handle_budget_exhaustion(ex_state, msg)
    assert handled_state.is_abstention is True
    assert handled_state.reason_code == ReasonCode.BUDGET_EXHAUSTED

    # Retrieval round budget hit (>2 rounds)
    round_ex_state = IndicRAGState(query_id="q3", question="test", llm_calls=2, retrieval_rounds=3)
    is_ex2, msg2 = supervisor.check_budget(round_ex_state)
    assert is_ex2 is True
    assert "Retrieval round budget exhausted" in msg2


def test_score_gate_s1_behavior():
    """Verify ScoreGate rejects low-score retrieval and admits high-score retrieval."""
    gate = ScoreGate(score_threshold=0.05)

    # Case 1: Low scores -> abstention
    low_state = IndicRAGState(
        query_id="q1",
        question="test",
        retrieved_passages=[{"passage_id": "p1", "text": "foo"}],
        retrieval_scores={"p1": 0.01},
    )
    res_low = gate.run(low_state)
    assert res_low.is_abstention is True
    assert res_low.reason_code == ReasonCode.NO_EVIDENCE

    # Case 2: High score -> passes
    high_state = IndicRAGState(
        query_id="q2",
        question="test",
        retrieved_passages=[{"passage_id": "p2", "text": "bar"}],
        retrieval_scores={"p2": 0.08},
    )
    res_high = gate.run(high_state)
    assert res_high.is_abstention is False


def test_evidence_grader_filtering_and_abstention():
    """Verify EvidenceGrader separates relevant passages from distractors and handles empty sets."""
    grader = EvidenceGraderAgent(relevance_threshold=0.20)

    passages = [
        {"passage_id": "p1", "title": "भारत", "text": "नई दिल्ली भारत की राजधानी है।", "score": 0.05},
        {"passage_id": "p2", "title": "क्रिकेट", "text": "सचिन तेंदुलकर महान बल्लेबाज हैं।", "score": 0.001},
    ]

    state = IndicRAGState(
        query_id="q1",
        question="भारत की राजधानी क्या है?",
        retrieved_passages=passages,
    )
    state = grader.run(state)

    # p1 should pass, p2 should be filtered out
    assert len(state.graded_passages) == 1
    assert state.graded_passages[0]["passage_id"] == "p1"
    assert state.is_abstention is False

    # Case where no passages pass
    noise_only_state = IndicRAGState(
        query_id="q2",
        question="भारत का अंतरिक्ष कार्यक्रम",
        retrieved_passages=[{"passage_id": "p_noise", "title": "फुटबॉल", "text": "मेस्सी ने गोल किया।", "score": 0.001}],
    )
    res_noise = grader.run(noise_only_state)
    assert len(res_noise.graded_passages) == 0
    assert res_noise.is_abstention is True
    assert res_noise.reason_code == ReasonCode.NO_EVIDENCE


def test_verifier_grounding_and_citation_fidelity():
    """Verify Verifier catches citation hallucinations, unsupported answers, and valid support."""
    from indicrag.agents.verifier_agent import VerifierAgent

    verifier = VerifierAgent(support_threshold=0.75)
    passages = [
        {"passage_id": "psg_001", "text": "नई दिल्ली भारत की आधिकारिक राजधानी है।"},
    ]

    # 1. Valid Grounded Answer with valid citation
    state_valid = IndicRAGState(
        query_id="q1",
        question="भारत की राजधानी क्या है?",
        draft_answer="[psg_001] नई दिल्ली भारत की राजधानी है।",
        cited_passage_ids=["psg_001"],
        graded_passages=passages,
        system_ladder=SystemLadderConfig.S2,
    )
    state_valid = verifier.run(state_valid)
    assert state_valid.verification_verdict == VerificationVerdict.SUPPORTED
    assert state_valid.is_abstention is False

    # 2. Citation Hallucination (citing non-existent passage)
    state_fake_cite = IndicRAGState(
        query_id="q2",
        question="भारत की राजधानी क्या है?",
        draft_answer="[psg_999] नई दिल्ली भारत की राजधानी है।",
        cited_passage_ids=["psg_999"],
        graded_passages=passages,
        system_ladder=SystemLadderConfig.S2,
    )
    state_fake_cite = verifier.run(state_fake_cite)
    assert state_fake_cite.verification_verdict == VerificationVerdict.UNSUPPORTED
    assert state_fake_cite.is_abstention is True
    assert state_fake_cite.reason_code == ReasonCode.UNSUPPORTED_ANSWER

    # 3. Unsupported Answer (claim not in evidence text)
    state_unsupported = IndicRAGState(
        query_id="q3",
        question="भारत की राजधानी क्या है?",
        draft_answer="[psg_001] मुम्बई भारत की राजधानी है।",
        cited_passage_ids=["psg_001"],
        graded_passages=passages,
        system_ladder=SystemLadderConfig.S2,
    )
    state_unsupported = verifier.run(state_unsupported)
    assert state_unsupported.verification_verdict == VerificationVerdict.UNSUPPORTED
    assert state_unsupported.is_abstention is True


def test_abstention_arbiter_bounded_retry_and_termination():
    """Verify Arbiter triggers retry on first verification failure and terminates if retries exhausted."""
    arbiter = AbstentionArbiter(max_retries=1)

    # Round 1: Unsupported answer -> trigger bounded retry
    state_retry = IndicRAGState(
        query_id="q1",
        question="test",
        verification_verdict=VerificationVerdict.UNSUPPORTED,
        retry_count=0,
        retrieval_rounds=1,
        llm_calls=2,
    )
    state_retry = arbiter.run(state_retry)
    assert getattr(state_retry, "_needs_retry", False) is True
    assert state_retry.retry_count == 1
    assert state_retry.is_abstention is False

    # Round 2: Still unsupported -> retries exhausted, finalize abstention
    state_retry.verification_verdict = VerificationVerdict.UNSUPPORTED
    state_final = arbiter.run(state_retry)
    assert getattr(state_final, "_needs_retry", False) is False
    assert state_final.is_abstention is True
    assert state_final.reason_code == ReasonCode.UNSUPPORTED_ANSWER


def test_system_ladder_execution_s0_through_s5():
    """Verify end-to-end execution of all systems ladder configurations (S0 through S5)."""
    retriever_func, passage_lookup = create_stub_retriever()
    stub_generator = StubAnswererGenerator()

    # Test each ladder configuration independently
    for ladder in [
        SystemLadderConfig.S0,
        SystemLadderConfig.S1,
        SystemLadderConfig.S2,
        SystemLadderConfig.S3,
        SystemLadderConfig.S4,
        SystemLadderConfig.S5,
    ]:
        graph = build_indicrag_graph(
            system_ladder=ladder,
            retriever_func=retriever_func,
            passage_lookup=passage_lookup,
            answerer_generator=stub_generator,
        )

        initial_state = IndicRAGState(
            query_id=f"test_{ladder.value}",
            question="भारत की राजधानी क्या है?",
            system_ladder=ladder,
        )

        final_state_dict = graph.invoke(initial_state)
        # LangGraph returns dict or state object
        history = final_state_dict.get("agent_history", []) if isinstance(final_state_dict, dict) else final_state_dict.agent_history
        final_answer = final_state_dict.get("final_answer") if isinstance(final_state_dict, dict) else final_state_dict.final_answer

        assert len(history) > 0, f"History empty for {ladder.value}"
        assert final_answer is not None, f"Final answer None for {ladder.value}"

        # Verify specific agent nodes per configuration
        if ladder == SystemLadderConfig.S0:
            assert "answerer_agent:generated" in history
            assert not any("verifier" in h for h in history)
        elif ladder == SystemLadderConfig.S1:
            assert any("score_gate" in h for h in history)
        elif ladder == SystemLadderConfig.S2:
            assert any("verifier" in h for h in history)
        elif ladder == SystemLadderConfig.S3:
            assert any("evidence_grader" in h for h in history)
        elif ladder == SystemLadderConfig.S4:
            assert "query_agent" in history
        elif ladder == SystemLadderConfig.S5:
            assert any("abstention_arbiter" in h for h in history)


def test_judge_agent_offline_isolation_in_graph():
    """Verify that JudgeAgent cannot be invoked inside the serving pipeline (AGENTS.md Rule 14)."""
    judge = JudgeAgent()
    with pytest.raises(RuntimeError, match="CRITICAL VIOLATION"):
        judge.evaluate_item(
            query_id="q_viol",
            question="what is this?",
            prediction="answer",
            is_impossible=False,
            gold_answers=["answer"],
            retrieved_passages=[],
            is_serving_pipeline=True,
        )


def test_no_gold_answers_leakage_in_state():
    """Verify that IndicRAGState contains zero evaluation labels or gold answer fields."""
    state_fields = IndicRAGState.model_fields.keys()
    forbidden_fields = {"gold_answer", "gold_answers", "is_impossible", "label", "human_label"}
    for f in forbidden_fields:
        assert f not in state_fields, f"Leakage violation: '{f}' present in serving IndicRAGState!"
