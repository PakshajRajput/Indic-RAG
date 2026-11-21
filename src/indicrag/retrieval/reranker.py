"""Cross-Encoder Reranker (R3) for PJ-09 IndicRAG-QA.

Reranks candidate passages using full cross-attention between query and passage.
Supports batch inference on CPU, score caching, and memory safety.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence

from sentence_transformers import CrossEncoder

os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class CrossEncoderReranker:
    """Cross-encoder reranker for candidates from R1 (BM25) or R2 (Dense)."""

    def __init__(
        self,
        model_name_or_path: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        batch_size: int = 16,
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        """Initialize CrossEncoder reranker.
        
        Args:
            model_name_or_path: Model name or checkpoint path.
            batch_size: Batch size for CPU scoring.
            device: Compute device ('cpu').
            max_length: Max token sequence length for cross-attention.
        """
        self.model_name = model_name_or_path
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self._model: CrossEncoder | None = None
        self._score_cache: dict[str, float] = {}

    @property
    def model(self) -> CrossEncoder:
        """Lazy loader for CrossEncoder model."""
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
        return self._model

    def _pair_cache_key(self, query: str, passage_text: str) -> str:
        """Stable hash key for query-passage pair."""
        return hashlib.sha256(f"{query}:::{passage_text}".encode()).hexdigest()[:16]

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, float]],
        passage_lookup: dict[str, str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Rerank candidates based on cross-encoder similarity.
        
        Args:
            query: Query string.
            candidates: Sequence of (passage_id, initial_score) tuples from R1/R2/R4.
            passage_lookup: Dict mapping passage_id to passage text (including title if desired).
            top_k: Number of reranked candidates to return.
            
        Returns:
            List of (passage_id, cross_encoder_score) sorted descending by score.
        """
        if not candidates:
            return []

        pairs_to_score: list[tuple[str, str]] = []
        pids_to_score: list[str] = []
        cached_results: dict[str, float] = {}

        for pid, _ in candidates:
            text = passage_lookup.get(pid, "")
            cache_key = self._pair_cache_key(query, text)
            if cache_key in self._score_cache:
                cached_results[pid] = self._score_cache[cache_key]
            else:
                pairs_to_score.append((query, text))
                pids_to_score.append(pid)

        # Predict for uncached pairs
        if pairs_to_score:
            scores = self.model.predict(
                pairs_to_score,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            for pid, pair, score in zip(pids_to_score, pairs_to_score, scores):
                s_float = float(score)
                self._score_cache[self._pair_cache_key(pair[0], pair[1])] = s_float
                cached_results[pid] = s_float

        # Sort candidates by cross-encoder score
        reranked = [
            (pid, cached_results.get(pid, -999.0))
            for pid, _ in candidates
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def batch_rerank(
        self,
        queries: Sequence[str],
        candidates_list: Sequence[Sequence[tuple[str, float]]],
        passage_lookup: dict[str, str],
        top_k: int = 10,
    ) -> list[list[tuple[str, float]]]:
        """Batch rerank multiple query candidate sets."""
        return [
            self.rerank(q, cands, passage_lookup, top_k=top_k)
            for q, cands in zip(queries, candidates_list)
        ]
