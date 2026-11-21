"""Evaluation Reporter and Per-Query Trace Exporter for PJ-09 IndicRAG-QA (Phase 7).

Provides:
- QueryRecord: Typed Pydantic container for granular per-query execution and evaluation outputs
- EvaluationReporter: Generates reproducible aggregate evaluation reports from per-query records
  with explicit denominators and guaranteed RGB metric identities:
    Acc_ans + FRR + Err_ans == 1.0
    Acc_unans + HR == 1.0
- Deterministic JSON/JSONL/CSV exports with complete agent and graph trace histories
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from indicrag.eval.metrics import compute_exact_match, compute_token_f1


class QueryRecord(BaseModel):
    """Detailed record for an individual query evaluation under a specific system."""
    query_id: str
    system_id: str
    benchmark_split: str  # 'dev' or 'test'
    corpus_condition: str  # 'clean' or 'perturbed'
    is_impossible: bool
    gold_answers: list[str] = Field(default_factory=list)
    predicted_text: str = ""
    is_abstention: bool = False
    reason_code: str | None = None
    cited_passage_ids: list[str] = Field(default_factory=list)
    retrieved_passage_ids: list[str] = Field(default_factory=list)
    retrieval_rounds: int = 1
    retry_count: int = 0
    is_correct: bool = False
    is_grounded: bool = True
    classification: str = ""  # 'correct_answer', 'false_refusal', 'incorrect_answer', 'correct_abstention', 'hallucination'
    token_f1: float = 0.0
    exact_match: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    is_cache_hit: bool = False
    traces: list[dict[str, Any]] = Field(default_factory=list)


class AggregateReport(BaseModel):
    """Aggregate evaluation report with explicit sample counts, denominators, and RGB identities."""
    system_id: str
    benchmark_split: str
    corpus_condition: str
    total_queries: int
    n_answerable: int
    n_unanswerable: int

    # Raw counts
    c_ans: int = Field(description="Correct answers on answerable queries")
    r_ans: int = Field(description="False refusals on answerable queries")
    w_ans: int = Field(description="Incorrect answers on answerable queries")
    c_unans: int = Field(description="Correct abstentions on unanswerable queries")
    h_unans: int = Field(description="Hallucinations on unanswerable queries")

    # Primary rates
    acc_ans: float = Field(description="Answerable accuracy (c_ans / n_answerable)")
    false_refusal_rate: float = Field(description="False refusal rate (r_ans / n_answerable)")
    error_rate_ans: float = Field(description="Incorrect answer rate (w_ans / n_answerable)")
    acc_unans: float = Field(description="Unanswerable accuracy (c_unans / n_unanswerable)")
    hallucination_rate: float = Field(description="Hallucination rate (h_unans / n_unanswerable)")
    acc_overall: float = Field(description="Overall benchmark accuracy (c_ans + c_unans) / total")
    rejection_rate: float = Field(description="Total rejection rate (r_ans + c_unans) / total")

    # RGB metric identities check
    ans_identity_holds: bool = Field(description="acc_ans + false_refusal_rate + error_rate_ans == 1.0")
    unans_identity_holds: bool = Field(description="acc_unans + hallucination_rate == 1.0")

    # QA and Citation metrics
    mean_token_f1: float = 0.0
    mean_exact_match: float = 0.0
    citation_precision: float = 0.0
    citation_recall: float = 0.0

    # Efficiency and Cache metrics
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    mean_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0


class EvaluationReporter:
    """Manages recording, export, and reproducible synthesis of evaluation runs."""

    def __init__(self) -> None:
        self.records: list[QueryRecord] = []

    def add_record(self, record: QueryRecord) -> None:
        """Add an evaluated query record."""
        # Auto-compute classification if not explicitly populated
        if not record.classification:
            if record.is_impossible:
                record.classification = "correct_abstention" if record.is_abstention else "hallucination"
            else:
                if record.is_abstention:
                    record.classification = "false_refusal"
                elif record.is_correct:
                    record.classification = "correct_answer"
                else:
                    record.classification = "incorrect_answer"

        # Compute QA scores if answerable and non-abstaining
        if not record.is_impossible and not record.is_abstention and record.gold_answers:
            if record.exact_match == 0.0 and record.token_f1 == 0.0:
                record.exact_match = compute_exact_match(record.predicted_text, record.gold_answers)
                record.token_f1 = compute_token_f1(record.predicted_text, record.gold_answers)

        self.records.append(record)

    def export_records_jsonl(self, filepath: str | Path) -> None:
        """Export all query records with full trace history to JSON Lines."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")

    def export_records_csv(self, filepath: str | Path) -> None:
        """Export tabular summary of records to CSV (omitting heavy traces)."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "query_id", "system_id", "benchmark_split", "corpus_condition",
            "is_impossible", "is_abstention", "reason_code", "is_correct",
            "is_grounded", "classification", "retrieval_rounds", "retry_count",
            "token_f1", "exact_match", "latency_ms", "prompt_tokens", "completion_tokens"
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.records:
                row = {k: getattr(rec, k) for k in fieldnames}
                writer.writerow(row)

    def generate_aggregate_report(
        self,
        system_id: str | None = None,
        benchmark_split: str | None = None,
        corpus_condition: str | None = None,
    ) -> AggregateReport:
        """Compute an explicit aggregate report adhering strictly to RGB mathematical guarantees."""
        subset = self.records
        if system_id is not None:
            subset = [r for r in subset if r.system_id == system_id]
        if benchmark_split is not None:
            subset = [r for r in subset if r.benchmark_split == benchmark_split]
        if corpus_condition is not None:
            subset = [r for r in subset if r.corpus_condition == corpus_condition]

        total = len(subset)
        ans_items = [r for r in subset if not r.is_impossible]
        unans_items = [r for r in subset if r.is_impossible]

        n_ans = len(ans_items)
        n_unans = len(unans_items)

        c_ans = sum(1 for r in ans_items if r.classification == "correct_answer")
        r_ans = sum(1 for r in ans_items if r.classification == "false_refusal")
        w_ans = sum(1 for r in ans_items if r.classification == "incorrect_answer")

        c_unans = sum(1 for r in unans_items if r.classification == "correct_abstention")
        h_unans = sum(1 for r in unans_items if r.classification == "hallucination")

        acc_ans = (c_ans / n_ans) if n_ans > 0 else 0.0
        frr = (r_ans / n_ans) if n_ans > 0 else 0.0
        err_ans = (w_ans / n_ans) if n_ans > 0 else 0.0

        acc_unans = (c_unans / n_unans) if n_unans > 0 else 0.0
        hr = (h_unans / n_unans) if n_unans > 0 else 0.0

        acc_overall = ((c_ans + c_unans) / total) if total > 0 else 0.0
        rejection_rate = ((r_ans + c_unans) / total) if total > 0 else 0.0

        # Mathematical RGB identities check
        ans_identity = (abs((acc_ans + frr + err_ans) - 1.0) < 1e-6) if n_ans > 0 else True
        unans_identity = (abs((acc_unans + hr) - 1.0) < 1e-6) if n_unans > 0 else True

        # QA metrics
        f1_list = [r.token_f1 for r in ans_items]
        em_list = [r.exact_match for r in ans_items]
        mean_f1 = (sum(f1_list) / len(f1_list)) if f1_list else 0.0
        mean_em = (sum(em_list) / len(em_list)) if em_list else 0.0

        # Latency and Cache
        lat_list = [r.latency_ms for r in subset]
        mean_lat = (sum(lat_list) / len(lat_list)) if lat_list else 0.0
        cache_hits = sum(1 for r in subset if r.is_cache_hit)
        hit_rate = (cache_hits / total) if total > 0 else 0.0

        return AggregateReport(
            system_id=system_id or "aggregate",
            benchmark_split=benchmark_split or "all",
            corpus_condition=corpus_condition or "all",
            total_queries=total,
            n_answerable=n_ans,
            n_unanswerable=n_unans,
            c_ans=c_ans,
            r_ans=r_ans,
            w_ans=w_ans,
            c_unans=c_unans,
            h_unans=h_unans,
            acc_ans=acc_ans,
            false_refusal_rate=frr,
            error_rate_ans=err_ans,
            acc_unans=acc_unans,
            hallucination_rate=hr,
            acc_overall=acc_overall,
            rejection_rate=rejection_rate,
            ans_identity_holds=ans_identity,
            unans_identity_holds=unans_identity,
            mean_token_f1=mean_f1,
            mean_exact_match=mean_em,
            total_prompt_tokens=sum(r.prompt_tokens for r in subset),
            total_completion_tokens=sum(r.completion_tokens for r in subset),
            mean_latency_ms=mean_lat,
            cache_hit_rate=hit_rate,
        )
