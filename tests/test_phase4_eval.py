"""Unit and integration test suite for Phase 4: Evaluation Harness, RGB Metrics, and Judge Agent.

Verifies:
1. Exact RGB metric calculations with explicit mathematical denominators
2. Zero-division safety under edge cases (empty list, 100% refusal, 0% refusal)
3. Offline Judge Agent model family independence (must reject identical family)
4. Offline Judge Agent serving pipeline isolation (must throw if invoked at serving time)
5. Pydantic typed schema validation for JudgeVerdict
6. Exact Cohen's Kappa score on controlled ratings
7. Stratification and audit integrity of 30-query sample (15 ans, 15 unans)
8. CostTracker token and dollar accounting with cache distinction
9. Zero leakage into Test partition (Dev benchmark only)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.indicrag.eval.cost_tracker import CostTracker
from src.indicrag.eval.human_audit import (
    calculate_cohens_kappa,
    interpret_kappa,
    run_stratified_human_audit,
)
from src.indicrag.eval.judge import (
    JudgeAgent,
    JudgeVerdict,
    VerdictCategory,
)
from src.indicrag.eval.metrics import (
    EvaluationItem,
    evaluate_rgb_metrics,
)


def test_rgb_metrics_mathematical_consistency():
    """Verify exact formula relationships: Acc_ans + FRR + Err_ans = 1.0, Acc_unans + HR = 1.0."""
    # Construct 10 answerable items:
    # 8 correct, 1 false refusal, 1 wrong
    ans_items = [
        EvaluationItem(
            query_id=f"a_{i}",
            is_impossible=False,
            predicted_text="उत्तर",
            is_abstention=False,
            is_correct=True,
            gold_answers=["उत्तर"],
        )
        for i in range(8)
    ] + [
        EvaluationItem(
            query_id="a_8",
            is_impossible=False,
            predicted_text="जानकारी नहीं है",
            is_abstention=True,
            is_correct=False,
            gold_answers=["उत्तर"],
        ),
        EvaluationItem(
            query_id="a_9",
            is_impossible=False,
            predicted_text="गलत उत्तर",
            is_abstention=False,
            is_correct=False,
            gold_answers=["उत्तर"],
        ),
    ]

    # Construct 10 unanswerable items:
    # 7 correct abstentions, 3 hallucinations
    unans_items = [
        EvaluationItem(
            query_id=f"u_{i}",
            is_impossible=True,
            predicted_text="जानकारी नहीं है",
            is_abstention=True,
            is_correct=True,
            gold_answers=[],
        )
        for i in range(7)
    ] + [
        EvaluationItem(
            query_id=f"u_{i}",
            is_impossible=True,
            predicted_text="मनगढ़ंत उत्तर",
            is_abstention=False,
            is_correct=False,
            gold_answers=[],
        )
        for i in range(7, 10)
    ]

    report = evaluate_rgb_metrics(ans_items + unans_items)

    assert report.total_queries == 20
    assert report.n_answerable == 10
    assert report.n_unanswerable == 10

    # Answerable: 8/10, 1/10, 1/10
    assert report.acc_ans == 0.8
    assert report.false_refusal_rate == 0.1
    assert report.error_rate_ans == 0.1
    assert abs(report.acc_ans + report.false_refusal_rate + report.error_rate_ans - 1.0) < 1e-6

    # Unanswerable: 7/10, 3/10
    assert report.acc_unans == 0.7
    assert report.hallucination_rate == 0.3
    assert abs(report.acc_unans + report.hallucination_rate - 1.0) < 1e-6

    # Overall accuracy: (8 + 7) / 20 = 15/20 = 0.75
    assert report.acc_overall == 0.75

    # Rejection rate: (1 refusal on ans + 7 abstentions on unans) / 20 = 8/20 = 0.40
    assert report.rejection_rate == 0.40


def test_rgb_metrics_zero_division_safety():
    """Verify edge cases (empty list, answerable-only, unanswerable-only) do not crash."""
    empty_rep = evaluate_rgb_metrics([])
    assert empty_rep.total_queries == 0
    assert empty_rep.acc_ans == 0.0
    assert empty_rep.acc_unans == 0.0

    ans_only = [
        EvaluationItem(
            query_id="a1",
            is_impossible=False,
            predicted_text="test",
            is_abstention=False,
            is_correct=True,
            gold_answers=["test"],
        )
    ]
    rep_ans = evaluate_rgb_metrics(ans_only)
    assert rep_ans.acc_ans == 1.0
    assert rep_ans.acc_unans == 0.0
    assert rep_ans.hallucination_rate == 0.0

    unans_only = [
        EvaluationItem(
            query_id="u1",
            is_impossible=True,
            predicted_text="refusal",
            is_abstention=True,
            is_correct=True,
            gold_answers=[],
        )
    ]
    rep_unans = evaluate_rgb_metrics(unans_only)
    assert rep_unans.acc_ans == 0.0
    assert rep_unans.acc_unans == 1.0
    assert rep_unans.false_refusal_rate == 0.0


def test_judge_agent_model_family_independence():
    """Enforce AGENTS.md Rule 14: Judge model family must be distinct from Answerer."""
    # Same family (Llama vs Llama) must fail
    with pytest.raises(ValueError, match="must be distinct"):
        JudgeAgent(
            judge_model_id="meta-llama/Llama-3.1-8B-Instruct",
            answerer_model_id="meta-llama/Llama-3.2-3B-Instruct",
        )

    # Different family (Google Gemini vs Meta Llama) must succeed
    judge = JudgeAgent(
        judge_model_id="google/gemini-2.0-flash",
        answerer_model_id="meta-llama/Llama-3.1-8B-Instruct",
    )
    assert judge.judge_family == "google_gemini"
    assert judge.answerer_family == "meta_llama"


def test_judge_agent_serving_pipeline_isolation():
    """Verify architectural assertion that Judge cannot be invoked inside serving pipeline."""
    judge = JudgeAgent(
        judge_model_id="google/gemini-2.0-flash",
        answerer_model_id="meta-llama/Llama-3.1-8B-Instruct",
    )

    with pytest.raises(RuntimeError, match="CRITICAL VIOLATION"):
        judge.evaluate_item(
            query_id="q1",
            question="प्रश्न?",
            prediction="उत्तर",
            is_impossible=False,
            gold_answers=["उत्तर"],
            retrieved_passages=[],
            is_serving_pipeline=True,  # Illegal invocation
        )


def test_judge_verdict_pydantic_schema():
    """Verify structured Pydantic JudgeVerdict schema."""
    verdict = JudgeVerdict(
        query_id="q123",
        is_correct=True,
        is_grounded=True,
        is_abstention=False,
        confidence=0.95,
        reasoning="Predicted answer aligns with reference answer span.",
        cited_passage_ids=["psg_abc123"],
        verdict_category=VerdictCategory.CORRECT_ANSWER,
    )
    serialized = verdict.model_dump()
    assert serialized["query_id"] == "q123"
    assert serialized["verdict_category"] == "CORRECT_ANSWER"
    roundtrip = JudgeVerdict(**serialized)
    assert roundtrip.is_correct is True


def test_cohens_kappa_controlled_calculation():
    """Verify exact Cohen's Kappa formula against hand-calculated ground truth."""
    # 100% agreement -> kappa = 1.0
    r1 = ["A", "B", "A", "C"]
    r2 = ["A", "B", "A", "C"]
    p_obs, kappa = calculate_cohens_kappa(r1, r2)
    assert p_obs == 1.0
    assert kappa == 1.0
    assert interpret_kappa(kappa) == "Almost perfect agreement"

    # Known 2x2 matrix:
    # 20 items:
    # r1: 15 yes, 5 no
    # r2: 12 yes, 8 no (12 both yes, 3 r1 yes & r2 no, 5 both no)
    r1_sample = ["Y"] * 12 + ["Y"] * 3 + ["N"] * 5
    r2_sample = ["Y"] * 12 + ["N"] * 3 + ["N"] * 5
    p_obs, kappa = calculate_cohens_kappa(r1_sample, r2_sample)
    assert p_obs == 17 / 20  # 0.85
    assert kappa > 0.60  # Substantial agreement


def test_stratified_human_audit_sample_integrity():
    """Verify that human audit draws a balanced 30-query stratified sample from Dev."""
    benchmark_dev_path = "data/processed/benchmark_dev_150.json"
    if not Path(benchmark_dev_path).exists():
        pytest.skip("benchmark_dev_150.json not found")

    audited_items, report = run_stratified_human_audit(
        benchmark_dev_path=benchmark_dev_path,
        sample_size=30,
        seed=42,
    )

    assert report.sample_size == 30
    assert report.n_answerable == 15
    assert report.n_unanswerable == 15
    assert len(audited_items) == 30

    ans_count = sum(1 for it in audited_items if not it.is_impossible)
    unans_count = sum(1 for it in audited_items if it.is_impossible)
    assert ans_count == 15
    assert unans_count == 15

    # Check observed agreement bound
    assert 0.0 <= report.observed_agreement <= 1.0
    assert -1.0 <= report.cohens_kappa <= 1.0


def test_cost_tracker_token_and_dollar_accounting():
    """Verify token accumulation and $0 cost on cache hits."""
    tracker = CostTracker()

    # 1. Live call
    tracker.record_call(
        call_id="call_1",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        prompt_tokens=1000,
        completion_tokens=200,
        latency_ms=450.0,
        is_cache_hit=False,
    )

    # 2. Cache hit (must be $0.00 cost)
    tracker.record_call(
        call_id="call_2",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        prompt_tokens=1000,
        completion_tokens=200,
        latency_ms=0.0,
        is_cache_hit=True,
    )

    summary = tracker.summary()
    assert summary["total_calls"] == 2
    assert summary["cache_hits"] == 1
    assert summary["live_calls"] == 1
    assert summary["cache_hit_rate"] == 0.5
    assert summary["total_tokens"] == 2400

    # Cost should strictly reflect only the 1 live call
    cost_1 = tracker.calculate_call_cost("meta-llama/Llama-3.1-8B-Instruct", 1000, 200, is_cache_hit=False)
    assert summary["total_cost_usd"] == pytest.approx(cost_1)


def test_evaluation_isolation_dev_only():
    """Verify zero test title or ID leakage into the Dev evaluation benchmark."""
    with open("data/processed/benchmark_dev_150.json", encoding="utf-8") as f:
        dev_bench = json.load(f)

    with open("data/processed/hindi_test.json", encoding="utf-8") as f:
        test_records = json.load(f)

    dev_titles = {q["title"] for q in dev_bench}
    dev_ids = {q["id"] for q in dev_bench}

    test_titles = {r["title"] for r in test_records}
    test_ids = {r["id"] for r in test_records}

    assert len(dev_titles & test_titles) == 0, "Test titles leaked into Dev benchmark!"
    assert len(dev_ids & test_ids) == 0, "Test question IDs leaked into Dev benchmark!"
