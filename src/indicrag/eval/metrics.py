"""RGB and RAG evaluation metrics module.

Implements exact metric calculations from Chen et al. (AAAI 2024, RGB benchmark):
- Acc_ans: Accuracy on answerable questions
- Acc_unans: Abstention accuracy on unanswerable questions (Negative Rejection)
- Acc_overall: Aggregate benchmark accuracy
- Rejection Rate (RR): Proportion of total questions where system abstained
- False Refusal Rate (FRR): Proportion of answerable questions mistakenly refused
- Hallucination / Overconfidence Rate (HR): Proportion of unanswerable questions where system fabricated an answer
- Citation Precision & Recall
- Exact Match (EM) and Indic-tokenized F1
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field


def normalize_text(text: str) -> str:
    """Normalize text for exact match and tokenization."""
    if not text:
        return ""
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    # Lowercase
    text = text.lower()
    # Strip punctuation and symbols (preserving Indic characters)
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    # Collapse multiple whitespace
    return " ".join(text.split()).strip()


def tokenize_indic(text: str) -> list[str]:
    """Tokenize text preserving Indic combining marks and alphanumeric words."""
    norm = normalize_text(text)
    if not norm:
        return []
    return re.findall(r"[\w\u0900-\u097F]+", norm)


def compute_exact_match(prediction: str, ground_truths: list[str]) -> float:
    """Compute exact match binary score against any valid ground truth."""
    if not ground_truths:
        return 0.0
    norm_pred = normalize_text(prediction)
    for gt in ground_truths:
        if norm_pred == normalize_text(gt):
            return 1.0
    return 0.0


def compute_token_f1(prediction: str, ground_truths: list[str]) -> float:
    """Compute token-level F1 score against ground truths (maximum over references)."""
    if not ground_truths:
        return 0.0
    pred_tokens = tokenize_indic(prediction)
    if not pred_tokens:
        return 0.0

    max_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = tokenize_indic(gt)
        if not gt_tokens:
            continue

        common = set(pred_tokens) & set(gt_tokens)
        num_same = sum(min(pred_tokens.count(tok), gt_tokens.count(tok)) for tok in common)
        if num_same == 0:
            f1 = 0.0
        else:
            precision = 1.0 * num_same / len(pred_tokens)
            recall = 1.0 * num_same / len(gt_tokens)
            f1 = (2.0 * precision * recall) / (precision + recall)
        if f1 > max_f1:
            max_f1 = f1

    return max_f1


class EvaluationItem(BaseModel):
    """Container for evaluating a single prediction item."""
    query_id: str
    is_impossible: bool  # True if query is unanswerable
    predicted_text: str
    is_abstention: bool  # True if system chose to abstain
    is_correct: bool     # Assessed by judge or exact/semantic evaluation
    gold_answers: list[str] = Field(default_factory=list)
    gold_passage_ids: list[str] = Field(default_factory=list)
    cited_passage_ids: list[str] = Field(default_factory=list)
    retrieved_passage_ids: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class RGBReport(BaseModel):
    """Aggregate evaluation report adhering strictly to RGB benchmark definitions."""
    total_queries: int
    n_answerable: int
    n_unanswerable: int

    # Accuracy metrics
    acc_ans: float = Field(description="Accuracy on answerable queries (C_ans / N_ans)")
    acc_unans: float = Field(description="Abstention accuracy on unanswerable queries (C_unans / N_unans)")
    acc_overall: float = Field(description="Overall benchmark accuracy (C_ans + C_unans) / N")

    # Abstention & error metrics
    rejection_rate: float = Field(description="Total rejection rate (Abstentions / N)")
    false_refusal_rate: float = Field(description="False refusal rate on answerable queries (R_ans / N_ans)")
    hallucination_rate: float = Field(description="Hallucination / overconfidence rate on unanswerable (H_unans / N_unans)")
    error_rate_ans: float = Field(description="Incorrect answers on answerable queries (W_ans / N_ans)")

    # QA metrics
    mean_exact_match: float = Field(description="Mean Exact Match on answerable queries")
    mean_token_f1: float = Field(description="Mean Token F1 on answerable queries")

    # Citation & grounding metrics
    mean_citation_precision: float = Field(description="Mean precision of cited passage IDs")
    mean_citation_recall: float = Field(description="Mean recall of cited passage IDs against gold passages")

    # Efficiency metrics
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    estimated_cost_usd: float = 0.0


def evaluate_rgb_metrics(items: list[EvaluationItem]) -> RGBReport:
    """Compute explicit RGB evaluation metrics across a list of EvaluationItems.

    Mathematical guarantees:
    - If N_ans == 0, answerable metrics return 0.0 without ZeroDivisionError.
    - If N_unans == 0, unanswerable metrics return 0.0 without ZeroDivisionError.
    - Acc_ans + FRR + Err_ans == 1.0 (within floating point precision) when N_ans > 0.
    - Acc_unans + HR == 1.0 when N_unans > 0.
    """
    total = len(items)
    if total == 0:
        return RGBReport(
            total_queries=0,
            n_answerable=0,
            n_unanswerable=0,
            acc_ans=0.0,
            acc_unans=0.0,
            acc_overall=0.0,
            rejection_rate=0.0,
            false_refusal_rate=0.0,
            hallucination_rate=0.0,
            error_rate_ans=0.0,
            mean_exact_match=0.0,
            mean_token_f1=0.0,
            mean_citation_precision=0.0,
            mean_citation_recall=0.0,
            mean_latency_ms=0.0,
            p95_latency_ms=0.0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            estimated_cost_usd=0.0,
        )

    ans_items = [it for it in items if not it.is_impossible]
    unans_items = [it for it in items if it.is_impossible]

    n_ans = len(ans_items)
    n_unans = len(unans_items)

    # 1. Answerable breakdown
    c_ans = 0
    r_ans = 0
    w_ans = 0
    em_scores = []
    f1_scores = []

    for it in ans_items:
        if it.is_abstention:
            r_ans += 1
            em_scores.append(0.0)
            f1_scores.append(0.0)
        elif it.is_correct:
            c_ans += 1
            em_scores.append(compute_exact_match(it.predicted_text, it.gold_answers))
            f1_scores.append(compute_token_f1(it.predicted_text, it.gold_answers))
        else:
            w_ans += 1
            em_scores.append(compute_exact_match(it.predicted_text, it.gold_answers))
            f1_scores.append(compute_token_f1(it.predicted_text, it.gold_answers))

    # 2. Unanswerable breakdown
    c_unans = 0
    h_unans = 0

    for it in unans_items:
        if it.is_abstention:
            c_unans += 1
        else:
            h_unans += 1

    acc_ans = (c_ans / n_ans) if n_ans > 0 else 0.0
    frr = (r_ans / n_ans) if n_ans > 0 else 0.0
    err_ans = (w_ans / n_ans) if n_ans > 0 else 0.0

    acc_unans = (c_unans / n_unans) if n_unans > 0 else 0.0
    hr = (h_unans / n_unans) if n_unans > 0 else 0.0

    acc_overall = (c_ans + c_unans) / total
    rejection_rate = (r_ans + c_unans) / total

    mean_em = (sum(em_scores) / len(em_scores)) if em_scores else 0.0
    mean_f1 = (sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0

    # 3. Citation metrics
    prec_list = []
    rec_list = []
    for it in ans_items:
        gold_set = set(it.gold_passage_ids)
        cited_set = set(it.cited_passage_ids)
        if cited_set:
            prec_list.append(len(cited_set & gold_set) / len(cited_set))
        if gold_set:
            rec_list.append(len(cited_set & gold_set) / len(gold_set))

    mean_cit_prec = (sum(prec_list) / len(prec_list)) if prec_list else 0.0
    mean_cit_rec = (sum(rec_list) / len(rec_list)) if rec_list else 0.0

    # 4. Latency and resource metrics
    latencies = [it.latency_ms for it in items]
    mean_lat = (sum(latencies) / len(latencies)) if latencies else 0.0

    if latencies:
        import numpy as np
        p95_lat = float(np.percentile(latencies, 95))
    else:
        p95_lat = 0.0

    prompt_toks = sum(it.prompt_tokens for it in items)
    comp_toks = sum(it.completion_tokens for it in items)

    return RGBReport(
        total_queries=total,
        n_answerable=n_ans,
        n_unanswerable=n_unans,
        acc_ans=acc_ans,
        acc_unans=acc_unans,
        acc_overall=acc_overall,
        rejection_rate=rejection_rate,
        false_refusal_rate=frr,
        hallucination_rate=hr,
        error_rate_ans=err_ans,
        mean_exact_match=mean_em,
        mean_token_f1=mean_f1,
        mean_citation_precision=mean_cit_prec,
        mean_citation_recall=mean_cit_rec,
        mean_latency_ms=mean_lat,
        p95_latency_ms=p95_lat,
        total_prompt_tokens=prompt_toks,
        total_completion_tokens=comp_toks,
        estimated_cost_usd=0.0,
    )
