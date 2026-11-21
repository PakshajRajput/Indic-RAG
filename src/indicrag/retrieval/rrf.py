"""Reciprocal Rank Fusion (R4) for PJ-09 IndicRAG-QA.

Combines multiple ranked lists (e.g., BM25 lexical + Dense multilingual)
without requiring raw score calibration or normalization.

Formula:
    RRF(d) = sum_{m in M} ( w_m / (c + rank_m(d)) )
where rank_m(d) is 1-based rank, c is a smoothing constant (default: 60).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[str, float]]],
    weights: Sequence[float] | None = None,
    c: int = 60,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists of (passage_id, score) using Reciprocal Rank Fusion.
    
    Args:
        ranked_lists: Sequence of ranked candidate lists, where each candidate is (passage_id, score).
        weights: Optional importance weights for each ranked list (must match len(ranked_lists)).
        c: Smoothing constant (default: 60 per Cormack et al., 2009).
        top_k: Number of fused results to return.
        
    Returns:
        List of (passage_id, rrf_score) sorted descending by rrf_score.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    elif len(weights) != len(ranked_lists):
        raise ValueError(
            f"Length of weights ({len(weights)}) must match length of ranked_lists ({len(ranked_lists)})"
        )

    fused_scores: dict[str, float] = defaultdict(float)

    for rank_list, weight in zip(ranked_lists, weights):
        for rank_idx, (pid, _) in enumerate(rank_list):
            # 1-based rank
            rank = rank_idx + 1
            fused_scores[pid] += weight / (c + rank)

    # Sort items by fused score descending
    sorted_items = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_items[:top_k]


class HybridRRFRetriever:
    """Wrapper that coordinates lexical (BM25) and dense retrieval with RRF fusion."""

    def __init__(
        self,
        retrievers: Sequence[Any],
        weights: Sequence[float] | None = None,
        c: int = 60,
        candidate_k: int = 20,
    ) -> None:
        """Initialize hybrid retriever.
        
        Args:
            retrievers: List of retriever objects (e.g. [bm25_retriever, dense_retriever]).
            weights: Optional weighting for each retriever.
            c: RRF smoothing constant (default: 60).
            candidate_k: Number of candidates each retriever should produce before fusion.
        """
        self.retrievers = list(retrievers)
        self.weights = weights
        self.c = c
        self.candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Retrieve top_k passages using hybrid RRF fusion."""
        ranked_lists = [
            r.retrieve(query, top_k=self.candidate_k)
            for r in self.retrievers
        ]
        return reciprocal_rank_fusion(ranked_lists, weights=self.weights, c=self.c, top_k=top_k)

    def batch_retrieve(
        self, queries: Sequence[str], top_k: int = 10
    ) -> list[list[tuple[str, float]]]:
        """Retrieve top_k passages for a batch of queries."""
        return [self.retrieve(q, top_k=top_k) for q in queries]
