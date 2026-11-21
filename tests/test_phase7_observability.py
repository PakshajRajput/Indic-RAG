"""Tests for Phase 7 Observability & Evaluation Pipeline.

Validates:
1. State propagation of explicit schema-backed needs_retry flag across LangGraph node boundaries.
2. End-to-end retry execution: retry produces a grounded, correctly evaluated final answer when evidence permits.
3. Bounded retry loop termination upon budget/retry exhaustion with typed reason codes.
4. Structured trace completeness across all agent nodes in the LangGraph serving pipeline.
5. Cache observability, telemetry, and isolation across system configurations.
6. RGB metric mathematical identities (Acc_ans + FRR + Err_ans == 1.0) and deterministic report generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from indicrag.agents.abstention_arbiter import AbstentionArbiter
from indicrag.agents.graph import build_indicrag_graph
from indicrag.agents.query_agent import QueryAgent
from indicrag.agents.state import (
    IndicRAGState,
    ReasonCode,
    ScriptType,
    SystemLadderConfig,
    VerificationVerdict,
)
from indicrag.agents.supervisor import SupervisorAgent
from indicrag.eval.reporter import AggregateReport, EvaluationReporter, QueryRecord
from indicrag.generation.generator import AnswererGenerator


def test_retry_flag_survives_graph_transition_and_propagates() -> None:
    """Validate that needs_retry survives Pydantic serialization across LangGraph nodes."""
    state = IndicRAGState(
        query_id="q_test_propagation",
        question="भारत की राजधानी क्या है?",
        system_ladder=SystemLadderConfig.S5,
        verification_verdict=VerificationVerdict.UNSUPPORTED,
        retry_count=0,
        max_retries=1,
        retrieval_rounds=1,
        max_retrieval_rounds=2,
    )

    arbiter = AbstentionArbiter(max_retries=1)
    state = arbiter.run(state)

    # 1. Flag must be explicitly set and schema-backed
    assert state.needs_retry is True
    assert state.retry_count == 1

    # 2. Test Supervisor routing recognizes the schema-backed flag
    supervisor = SupervisorAgent(max_retrieval_rounds=2)
    next_node = supervisor.route_after_arbiter(state)
    assert next_node == "query_agent"

    # 3. Test QueryAgent detects and resets the schema-backed flag
    query_agent = QueryAgent()
    state = query_agent.run(state)
    assert state.needs_retry is False
    assert state.active_query != ""
    assert "query_agent:retry_reformulation" in state.agent_history


def test_retry_produces_grounded_answer_when_evidence_permits() -> None:
    """Validate that retry produces a grounded, correctly cited answer when round 2 retrieves gold evidence."""
    passage_pool = {
        "distractor_1": {"passage_id": "distractor_1", "title": "Random", "text": "यहाँ कोई प्रासंगिक तथ्य नहीं है।", "score": 0.50},
        "gold_psg_1": {"passage_id": "gold_psg_1", "title": "राजधानी", "text": "भारत की राजधानी नई दिल्ली है।", "score": 0.90},
    }

    class MockRetriever:
        def __init__(self) -> None:
            self.round = 0

        def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
            self.round += 1
            if self.round == 1:
                return [("distractor_1", 0.50)]
            return [("gold_psg_1", 0.90)]

    class MockAnswererGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, query_id: str, question: str, retrieved_passages: list[dict[str, Any]], **kwargs: Any) -> Any:
            from indicrag.generation.generator import GenerationResult
            self.calls += 1
            if self.calls == 1:
                # Round 1: draft answer unsupported by distractor_1 text
                return GenerationResult(
                    query_id=query_id,
                    model_name="mock_model",
                    prompt="test",
                    response_text="भारत की राजधानी मुम्बई है। [distractor_1]",
                    is_abstention=False,
                    prompt_tokens=50,
                    completion_tokens=20,
                    latency_ms=5.0,
                    is_cache_hit=False,
                )
            # Round 2: grounded answer supported by gold_psg_1
            return GenerationResult(
                query_id=query_id,
                model_name="mock_model",
                prompt="test",
                response_text="भारत की राजधानी नई दिल्ली है। [gold_psg_1]",
                is_abstention=False,
                prompt_tokens=50,
                completion_tokens=20,
                latency_ms=5.0,
                is_cache_hit=False,
            )

    retriever_mock = MockRetriever()
    gen_mock = MockAnswererGenerator()

    app = build_indicrag_graph(
        system_ladder=SystemLadderConfig.S5,
        retriever_func=retriever_mock.retrieve,
        passage_lookup=passage_pool,
        answerer_generator=gen_mock,
        score_threshold=0.005,
        evidence_threshold=0.20,
        support_threshold=0.75,
        max_retries=1,
    )

    init_state = IndicRAGState(
        query_id="q_retry_recover",
        question="भारत की राजधानी क्या है?",
        system_ladder=SystemLadderConfig.S5,
    )

    out = app.invoke(init_state)
    final_out = out if isinstance(out, IndicRAGState) else IndicRAGState(**out)

    # Execution assertions: retry was triggered and recovered the answer
    assert final_out.retry_count == 1
    assert final_out.retrieval_rounds == 2
    assert final_out.is_abstention is False
    assert "नई दिल्ली" in (final_out.final_answer or "")
    assert "gold_psg_1" in final_out.cited_passage_ids
    assert "abstention_arbiter:accepted" in final_out.agent_history


def test_retry_terminates_deterministically_when_budget_exhausted() -> None:
    """Validate that retry does not loop indefinitely and terminates with typed abstention."""
    class UnsupportedRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
            return [("psg_noise", 0.50)]

    passage_pool = {
        "psg_noise": {"passage_id": "psg_noise", "title": "शोर", "text": "यहाँ सिर्फ शोर और असंबंधित शब्द हैं।", "score": 0.50}
    }

    class AlwaysUnsupportedGenerator:
        def generate(self, query_id: str, question: str, retrieved_passages: list[dict[str, Any]], **kwargs: Any) -> Any:
            from indicrag.generation.generator import GenerationResult
            # Always generates an answer claiming things not in the noise passage
            return GenerationResult(
                query_id=query_id,
                model_name="mock_model",
                prompt="test",
                response_text="मंगल ग्रह पर जल की खोज हुई है। [psg_noise]",
                is_abstention=False,
                prompt_tokens=50,
                completion_tokens=20,
                latency_ms=5.0,
                is_cache_hit=False,
            )

    app = build_indicrag_graph(
        system_ladder=SystemLadderConfig.S5,
        retriever_func=UnsupportedRetriever().retrieve,
        passage_lookup=passage_pool,
        answerer_generator=AlwaysUnsupportedGenerator(),
        score_threshold=0.005,
        evidence_threshold=0.20,
        support_threshold=0.75,
        max_retries=1,
        max_retrieval_rounds=2,
    )

    init_state = IndicRAGState(
        query_id="q_retry_exhaust",
        question="मंगल ग्रह के बारे में बताइए?",
        system_ladder=SystemLadderConfig.S5,
    )

    out = app.invoke(init_state)
    final_out = out if isinstance(out, IndicRAGState) else IndicRAGState(**out)

    assert final_out.is_abstention is True
    assert final_out.retry_count == 1
    assert final_out.retrieval_rounds == 2
    assert final_out.reason_code == ReasonCode.UNSUPPORTED_ANSWER
    assert f"abstention_arbiter:final_abstention_{ReasonCode.UNSUPPORTED_ANSWER.value}" in final_out.agent_history


def test_structured_trace_completeness_across_all_nodes() -> None:
    """Validate that state.traces captures granular records from all graph nodes."""
    passage_pool = {
        "psg_delhi": {"passage_id": "psg_delhi", "title": "राजधानी", "text": "भारत की राजधानी नई दिल्ली है।"}
    }

    gen = AnswererGenerator(seed=42)
    gen.register_evaluation_ground_truth([{
        "id": "q_trace_check",
        "question": "भारत की राजधानी क्या है?",
        "is_impossible": False,
        "answers": {"text": ["नई दिल्ली"]},
    }])

    app = build_indicrag_graph(
        system_ladder=SystemLadderConfig.S5,
        retriever_func=lambda q, k: [("psg_delhi", 0.08)],
        passage_lookup=passage_pool,
        answerer_generator=gen,
    )

    st = IndicRAGState(query_id="q_trace_check", question="भारत की राजधानी क्या है?", system_ladder=SystemLadderConfig.S5)
    out = app.invoke(st)
    final_out = out if isinstance(out, IndicRAGState) else IndicRAGState(**out)

    assert len(final_out.traces) >= 6
    node_names = [t["node"] for t in final_out.traces]
    assert "supervisor" in node_names
    assert "script_detector" in node_names
    assert "query_agent" in node_names
    assert "retriever" in node_names
    assert "answerer" in node_names
    assert "verifier" in node_names
    assert "abstention_arbiter" in node_names

    # Check that trace records contain round, retry count, and action
    for t in final_out.traces:
        assert "round" in t
        assert "retry_count" in t
        assert "action" in t


def test_cache_observability_and_audit_isolation(tmp_path: Path) -> None:
    """Validate cache hit/miss tracking and provenance metadata logging."""
    gen = AnswererGenerator(cache_dir=str(tmp_path / "cache"), seed=42)
    gen.register_evaluation_ground_truth([{
        "id": "q_cache_test",
        "question": "गंगा नदी कहाँ बहती है?",
        "is_impossible": False,
        "answers": {"text": ["भारत"]},
    }])

    passages = [{"passage_id": "psg_ganga", "title": "गंगा", "text": "गंगा नदी भारत में बहती है।"}]

    # 1. First call -> Cache Miss
    res1 = gen.generate(
        query_id="q_cache_test",
        question="गंगा नदी कहाँ बहती है?",
        retrieved_passages=passages,
        system_id="S5",
        corpus_condition="clean",
        prompt_version="1.0.0",
    )
    assert res1.is_cache_hit is False
    assert gen.total_cache_misses == 1
    assert gen.total_cache_hits == 0

    # 2. Second call with same inputs -> Cache Hit
    res2 = gen.generate(
        query_id="q_cache_test",
        question="गंगा नदी कहाँ बहती है?",
        retrieved_passages=passages,
        system_id="S5",
        corpus_condition="clean",
        prompt_version="1.0.0",
    )
    assert res2.is_cache_hit is True
    assert gen.total_cache_hits == 1

    stats = gen.get_cache_stats()
    assert stats["total_requests"] == 2
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5

    # Check audit log entries
    assert len(gen.cache_audit_log) == 2
    assert gen.cache_audit_log[0]["is_cache_hit"] is False
    assert gen.cache_audit_log[1]["is_cache_hit"] is True
    assert gen.cache_audit_log[1]["system_id"] == "S5"
    assert gen.cache_audit_log[1]["corpus_condition"] == "clean"


def test_rgb_metric_identities_and_deterministic_reporting(tmp_path: Path) -> None:
    """Validate that EvaluationReporter preserves mathematical RGB identities on aggregate reports."""
    reporter = EvaluationReporter()

    # Add 10 controlled items:
    # 6 Answerable: 4 correct, 1 false refusal, 1 incorrect
    # 4 Unanswerable: 3 correct abstentions, 1 hallucination
    items = [
        # Answerable correct
        QueryRecord(query_id="q1", system_id="S5", benchmark_split="dev", corpus_condition="perturbed", is_impossible=False, is_correct=True, is_abstention=False, classification="correct_answer"),
        QueryRecord(query_id="q2", system_id="S5", benchmark_split="dev", corpus_condition="perturbed", is_impossible=False, is_correct=True, is_abstention=False, classification="correct_answer"),
        QueryRecord(query_id="q3", system_id="S5", benchmark_split="dev", corpus_condition="perturbed", is_impossible=False, is_correct=True, is_abstention=False, classification="correct_answer"),
        QueryRecord(query_id="q4", system_id="S5", benchmark_split="dev", corpus_condition="perturbed", is_impossible=False, is_correct=True, is_abstention=False, classification="correct_answer"),
        # Answerable false refusal
        QueryRecord(query_id="q5", system_id="S5", benchmark_split="dev", corpus_condition="perturbed", is_impossible=False, is_correct=False, is_abstention=True, classification="false_refusal"),
        # Answerable incorrect answer
        QueryRecord(query_id="q6", system_id="S5", benchmark_split="dev", corpus_condition="perturbed", is_impossible=False, is_correct=False, is_abstention=False, classification="incorrect_answer"),
        # Unanswerable correct abstention
        QueryRecord(query_id="q7", system_id="S5", benchmark_split="dev", corpus_condition="perturbed", is_impossible=True, is_correct=True, is_abstention=True, classification="correct_abstention"),
        QueryRecord(query_id="q8", system_id="S5", benchmark_split="dev", corpus_condition="perturbed", is_impossible=True, is_correct=True, is_abstention=True, classification="correct_abstention"),
        QueryRecord(query_id="q9", system_id="S5", benchmark_split="dev", corpus_condition="perturbed", is_impossible=True, is_correct=True, is_abstention=True, classification="correct_abstention"),
        # Unanswerable hallucination
        QueryRecord(query_id="q10", system_id="S5", benchmark_split="dev", corpus_condition="perturbed", is_impossible=True, is_correct=False, is_abstention=False, classification="hallucination"),
    ]

    for it in items:
        reporter.add_record(it)

    rep: AggregateReport = reporter.generate_aggregate_report(system_id="S5")

    # Sample denominators
    assert rep.total_queries == 10
    assert rep.n_answerable == 6
    assert rep.n_unanswerable == 4

    # Raw counts
    assert rep.c_ans == 4
    assert rep.r_ans == 1
    assert rep.w_ans == 1
    assert rep.c_unans == 3
    assert rep.h_unans == 1

    # Exact RGB mathematical identities
    assert rep.ans_identity_holds is True
    assert rep.unans_identity_holds is True
    assert abs((rep.acc_ans + rep.false_refusal_rate + rep.error_rate_ans) - 1.0) < 1e-6
    assert abs((rep.acc_unans + rep.hallucination_rate) - 1.0) < 1e-6

    # Rates
    assert rep.acc_ans == 4.0 / 6.0
    assert rep.false_refusal_rate == 1.0 / 6.0
    assert rep.error_rate_ans == 1.0 / 6.0
    assert rep.acc_unans == 3.0 / 4.0
    assert rep.hallucination_rate == 1.0 / 4.0
    assert rep.acc_overall == (4 + 3) / 10.0

    # Test file exports
    jsonl_path = tmp_path / "traces_export.jsonl"
    csv_path = tmp_path / "records_export.csv"
    reporter.export_records_jsonl(jsonl_path)
    reporter.export_records_csv(csv_path)

    assert jsonl_path.exists()
    assert csv_path.exists()
    with open(jsonl_path, encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 10
