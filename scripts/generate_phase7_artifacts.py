"""Generate Phase 7 Observability & Evaluation Pipeline demonstration artifacts.

Exports:
- results/phase7_traces_sample.jsonl
- results/phase7_records_sample.csv
- results/phase7_aggregate_report.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indicrag.eval.reporter import EvaluationReporter, QueryRecord
from indicrag.generation.generator import AnswererGenerator


def main() -> None:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    reporter = EvaluationReporter()

    # Initialize generator to audit cache stats
    generator = AnswererGenerator(seed=42)

    # Synthetic sample records illustrating full lifecycle across S2, S4, S5
    sample_records = [
        QueryRecord(
            query_id="dev_ans_001",
            system_id="S5",
            benchmark_split="dev",
            corpus_condition="clean",
            is_impossible=False,
            gold_answers=["नई दिल्ली"],
            predicted_text="भारत की राजधानी नई दिल्ली है। [psg_delhi_01]",
            is_abstention=False,
            is_correct=True,
            is_grounded=True,
            classification="correct_answer",
            cited_passage_ids=["psg_delhi_01"],
            retrieved_passage_ids=["psg_delhi_01"],
            retrieval_rounds=1,
            retry_count=0,
            token_f1=1.0,
            exact_match=1.0,
            prompt_tokens=120,
            completion_tokens=65,
            latency_ms=120.5,
            reason_code=None,
            is_cache_hit=True,
            traces=[
                {"node": "supervisor", "action": "budget_ok", "round": 0},
                {"node": "script_detector", "action": "detected", "details": {"script": "devanagari"}},
                {"node": "query_agent", "action": "formulated"},
                {"node": "retriever", "action": "round_1", "round": 1},
                {"node": "evidence_grader", "action": "passed_passages", "details": {"count": 2}},
                {"node": "answerer", "action": "generated"},
                {"node": "verifier", "action": "SUPPORTED"},
                {"node": "abstention_arbiter", "action": "accepted"},
            ],
        ),
        QueryRecord(
            query_id="dev_ans_002",
            system_id="S5",
            benchmark_split="dev",
            corpus_condition="perturbed",
            is_impossible=False,
            gold_answers=["1947"],
            predicted_text="भारत 1947 में स्वतंत्र हुआ था। [psg_independence_01]",
            is_abstention=False,
            is_correct=True,
            is_grounded=True,
            classification="correct_answer",
            cited_passage_ids=["psg_independence_01"],
            retrieved_passage_ids=["psg_noise_01", "psg_independence_01"],
            retrieval_rounds=2,
            retry_count=1,
            token_f1=1.0,
            exact_match=1.0,
            prompt_tokens=250,
            completion_tokens=140,
            latency_ms=250.2,
            reason_code=None,
            is_cache_hit=False,
            traces=[
                {"node": "retriever", "action": "round_1", "round": 1},
                {"node": "verifier", "action": "UNSUPPORTED", "round": 1},
                {"node": "abstention_arbiter", "action": "trigger_retry", "details": {"retry_count": 1}},
                {"node": "query_agent", "action": "retry_reformulation", "round": 1},
                {"node": "retriever", "action": "round_2", "round": 2},
                {"node": "verifier", "action": "SUPPORTED", "round": 2},
                {"node": "abstention_arbiter", "action": "accepted", "round": 2},
            ],
        ),
        QueryRecord(
            query_id="dev_ans_003",
            system_id="S5",
            benchmark_split="dev",
            corpus_condition="perturbed",
            is_impossible=False,
            gold_answers=["आर्यभट्ट"],
            predicted_text="सत्यापन में दावे असमर्थित पाए गए, इसलिए उत्तर अस्वीकार किया जाता है।",
            is_abstention=True,
            is_correct=False,
            is_grounded=False,
            classification="false_refusal",
            cited_passage_ids=[],
            retrieved_passage_ids=["psg_noise_02"],
            retrieval_rounds=2,
            retry_count=1,
            token_f1=0.0,
            exact_match=0.0,
            prompt_tokens=230,
            completion_tokens=130,
            latency_ms=230.1,
            reason_code="UNSUPPORTED_ANSWER",
            is_cache_hit=False,
            traces=[
                {"node": "retriever", "action": "round_1", "round": 1},
                {"node": "verifier", "action": "UNSUPPORTED", "round": 1},
                {"node": "abstention_arbiter", "action": "trigger_retry", "details": {"retry_count": 1}},
                {"node": "query_agent", "action": "retry_reformulation", "round": 1},
                {"node": "retriever", "action": "round_2", "round": 2},
                {"node": "verifier", "action": "UNSUPPORTED", "round": 2},
                {"node": "abstention_arbiter", "action": "final_abstention_UNSUPPORTED_ANSWER", "round": 2},
            ],
        ),
        QueryRecord(
            query_id="dev_unans_001",
            system_id="S5",
            benchmark_split="dev",
            corpus_condition="perturbed",
            is_impossible=True,
            gold_answers=[],
            predicted_text="दिए गए संदर्भ में प्रश्न का उत्तर देने हेतु कोई साक्ष्य उपलब्ध नहीं है।",
            is_abstention=True,
            is_correct=False,
            is_grounded=True,
            classification="correct_abstention",
            cited_passage_ids=[],
            retrieved_passage_ids=["psg_distractor_03"],
            retrieval_rounds=1,
            retry_count=0,
            token_f1=0.0,
            exact_match=0.0,
            prompt_tokens=60,
            completion_tokens=30,
            latency_ms=50.0,
            reason_code="NO_EVIDENCE",
            is_cache_hit=True,
            traces=[
                {"node": "retriever", "action": "round_1"},
                {"node": "evidence_grader", "action": "all_passages_failed"},
                {"node": "abstention_arbiter", "action": "abstained_no_evidence"},
            ],
        ),
    ]

    for rec in sample_records:
        reporter.add_record(rec)

    # Generate and export
    # Generate aggregate report
    report = reporter.generate_aggregate_report(
        system_id="S5",
        benchmark_split="dev",
    )

    reporter.export_records_csv(results_dir / "phase7_records_sample.csv")
    reporter.export_records_jsonl(results_dir / "phase7_traces_sample.jsonl")
    
    report_file = results_dir / "phase7_aggregate_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    print(f"Generated {len(sample_records)} sample records.")
    print(f"Report exported to {report_file}")
    print(f"Acc_ans: {report.acc_ans:.4f}, FRR: {report.false_refusal_rate:.4f}, Err_ans: {report.error_rate_ans:.4f}")
    print(f"Unanswerable Acc: {report.acc_unans:.4f}")
    print(f"Cache Hit Rate: {report.cache_hit_rate:.4f}")
    print(f"RGB Identity holds: {report.ans_identity_holds}")


if __name__ == "__main__":
    main()
