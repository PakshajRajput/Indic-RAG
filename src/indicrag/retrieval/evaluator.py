"""IR Metrics Evaluator for Retrieval Pipeline (Phase 3).

Computes Recall@k, MRR@10, nDCG@10, and latency across query sets,
with proper handling of unanswerable queries (empty gold evidence).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any


def compute_dcg(relevances: Sequence[int], k: int = 10) -> float:
    """Compute Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            dcg += rel / math.log2(i + 2)  # i+2 because i is 0-indexed: rank 1 is log2(2)
    return dcg


def compute_ndcg_at_k(retrieved_pids: Sequence[str], gold_pids: set[str], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k with binary relevance."""
    if not gold_pids:
        return 0.0

    relevances = [1 if pid in gold_pids else 0 for pid in retrieved_pids[:k]]
    actual_dcg = compute_dcg(relevances, k=k)

    # Ideal DCG: all min(|gold_pids|, k) relevant documents at top
    ideal_count = min(len(gold_pids), k)
    ideal_relevances = [1] * ideal_count
    ideal_dcg = compute_dcg(ideal_relevances, k=k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_mrr_at_k(retrieved_pids: Sequence[str], gold_pids: set[str], k: int = 10) -> float:
    """Compute Reciprocal Rank at rank k."""
    if not gold_pids:
        return 0.0

    for i, pid in enumerate(retrieved_pids[:k]):
        if pid in gold_pids:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(retrieved_pids: Sequence[str], gold_pids: set[str], k: int) -> float:
    """Compute Recall@k (binary hit rate for QA gold passage)."""
    if not gold_pids:
        return 0.0

    retrieved_set = set(retrieved_pids[:k])
    hit = 1.0 if any(pid in retrieved_set for pid in gold_pids) else 0.0
    return hit


class RetrievalEvaluator:
    """Standardized retrieval evaluation runner for PJ-09 IndicRAG-QA."""

    def __init__(self, k_values: Sequence[int] = (1, 5, 10, 20)) -> None:
        self.k_values = list(k_values)

    def evaluate_retrieval(
        self,
        queries: Sequence[dict[str, Any]],
        gold_passage_map: dict[str, set[str]],
        retriever_func: Any,
        top_k: int = 20,
    ) -> dict[str, Any]:
        """Evaluate a retriever over a query set.
        
        Args:
            queries: List of query dicts (must have 'id' and 'question').
            gold_passage_map: Map from query 'id' to set of gold passage IDs.
            retriever_func: Callable (query_str, top_k) -> List[Tuple[passage_id, score]].
            top_k: Max candidate depth.
            
        Returns:
            Dictionary of metrics including Recall@k, MRR@10, nDCG@10, and latency.
        """
        recalls_at_k: dict[int, list[float]] = {k: [] for k in self.k_values}
        mrrs_at_10: list[float] = []
        ndcgs_at_10: list[float] = []
        latencies_ms: list[float] = []

        ans_count = 0
        unans_count = 0

        for q_item in queries:
            qid = str(q_item["id"])
            q_text = q_item["question"]
            gold_pids = gold_passage_map.get(qid, set())
            is_unans = bool(q_item.get("is_impossible", len(gold_pids) == 0))

            t0 = time.perf_counter()
            retrieved = retriever_func(q_text, top_k=top_k)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            retrieved_pids = [pid for pid, _ in retrieved]

            if is_unans or len(gold_pids) == 0:
                unans_count += 1
                # Unanswerable query has no gold evidence; skip IR recall/mrr/ndcg to avoid division by zero
                continue

            ans_count += 1
            for k in self.k_values:
                recalls_at_k[k].append(compute_recall_at_k(retrieved_pids, gold_pids, k))

            mrrs_at_10.append(compute_mrr_at_k(retrieved_pids, gold_pids, k=10))
            ndcgs_at_10.append(compute_ndcg_at_k(retrieved_pids, gold_pids, k=10))

        mean_metrics: dict[str, Any] = {
            "num_queries_total": len(queries),
            "num_answerable": ans_count,
            "num_unanswerable": unans_count,
            "mean_latency_ms": float(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0,
        }

        for k in self.k_values:
            vals = recalls_at_k[k]
            mean_metrics[f"recall@{k}"] = float(sum(vals) / len(vals)) if vals else 0.0

        mean_metrics["mrr@10"] = float(sum(mrrs_at_10) / len(mrrs_at_10)) if mrrs_at_10 else 0.0
        mean_metrics["ndcg@10"] = float(sum(ndcgs_at_10) / len(ndcgs_at_10)) if ndcgs_at_10 else 0.0

        return mean_metrics
