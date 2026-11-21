"""Dense Multilingual Retriever (R2) using SentenceTransformers with persistent disk caching.

Supports multilingual embedding models (e.g., paraphrase-multilingual-MiniLM-L12-v2,
intfloat/multilingual-e5-small, intfloat/multilingual-e5-base, BAAI/bge-m3).
Caches corpus embeddings to disk to prevent redundant CPU computation.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure Hugging Face cache stays on D: drive
os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class DenseRetriever:
    """Dense embedding retriever (R2) over a searchable passage corpus."""

    def __init__(
        self,
        corpus: Sequence[dict[str, Any]],
        model_name_or_path: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir: str = "data/processed/embeddings",
        corpus_id: str = "default",
        batch_size: int = 16,
        device: str = "cpu",
        include_title: bool = True,
    ) -> None:
        """Initialize Dense Retriever with corpus and embedding model.
        
        Args:
            corpus: List of passage dictionaries (passage_id, text, title).
            model_name_or_path: Hugging Face model identifier or local path.
            cache_dir: Directory where precomputed corpus embeddings are saved.
            corpus_id: Distinct identifier for the corpus (e.g. 'dev_clean', 'dev_perturbed').
            batch_size: Batch size for CPU embedding encoding.
            device: Compute device ('cpu').
            include_title: Whether to prepend passage title to passage text.
        """
        self.model_name = model_name_or_path
        self.cache_dir = Path(cache_dir)
        self.corpus_id = corpus_id
        self.batch_size = batch_size
        self.device = device
        self.include_title = include_title
        self.corpus = corpus

        # Prefixes for e5 family models
        self.is_e5 = "e5" in model_name_or_path.lower()
        self.passage_prefix = "passage: " if self.is_e5 else ""
        self.query_prefix = "query: " if self.is_e5 else ""

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.passage_ids: list[str] = []
        self.passage_embeddings: np.ndarray | None = None
        self._model: SentenceTransformer | None = None

        self._initialize_index()

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader for SentenceTransformer model to manage memory."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _get_cache_paths(self) -> tuple[Path, Path]:
        """Compute stable hash for model and corpus to identify cache files."""
        # Create hash based on model name, corpus_id, count, and sample passage IDs
        p_ids = [str(item.get("passage_id", item.get("id", ""))) for item in self.corpus]
        hash_input = f"{self.model_name}::{self.corpus_id}::{len(p_ids)}::{p_ids[:10]}::{p_ids[-10:]}"
        cache_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]

        safe_model_name = self.model_name.replace("/", "_").replace("\\", "_")
        emb_path = self.cache_dir / f"{safe_model_name}_{self.corpus_id}_{cache_hash}_emb.npy"
        ids_path = self.cache_dir / f"{safe_model_name}_{self.corpus_id}_{cache_hash}_ids.json"
        return emb_path, ids_path

    def _initialize_index(self) -> None:
        """Load embeddings from disk cache or compute and save them."""
        emb_path, ids_path = self._get_cache_paths()

        if emb_path.exists() and ids_path.exists():
            with open(ids_path, encoding="utf-8") as f:
                self.passage_ids = json.load(f)
            self.passage_embeddings = np.load(str(emb_path))
            return

        # Prepare corpus texts
        self.passage_ids = []
        texts_to_encode: list[str] = []

        for item in self.corpus:
            pid = str(item.get("passage_id", item.get("id", "")))
            title = item.get("title", "")
            text = item.get("text", item.get("context", ""))

            if self.include_title and title:
                full_text = f"{title} {text}"
            else:
                full_text = text

            formatted_text = f"{self.passage_prefix}{full_text}"
            self.passage_ids.append(pid)
            texts_to_encode.append(formatted_text)

        if not texts_to_encode:
            self.passage_embeddings = np.empty((0, 384), dtype=np.float32)
            return

        # Compute embeddings in batches
        embeddings = self.model.encode(
            texts_to_encode,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self.passage_embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Save to disk cache
        np.save(str(emb_path), self.passage_embeddings)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(self.passage_ids, f, ensure_ascii=False)

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Retrieve top_k passages for a single query.
        
        Args:
            query: Input question or query string.
            top_k: Number of candidate passages.
            
        Returns:
            List of (passage_id, score) sorted descending by similarity.
        """
        if self.passage_embeddings is None or len(self.passage_ids) == 0:
            return []

        formatted_query = f"{self.query_prefix}{query}"
        q_emb = self.model.encode(
            formatted_query,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self.passage_embeddings, q_emb)

        top_k = min(top_k, len(self.passage_ids))
        if top_k <= 0:
            return []

        # Sort indices by score descending
        ranked_indices = np.argsort(-scores)[:top_k]
        return [(self.passage_ids[i], float(scores[i])) for i in ranked_indices]

    def batch_retrieve(
        self, queries: Sequence[str], top_k: int = 10
    ) -> list[list[tuple[str, float]]]:
        """Retrieve top_k passages for a batch of queries."""
        if self.passage_embeddings is None or len(self.passage_ids) == 0:
            return [[] for _ in queries]

        formatted_queries = [f"{self.query_prefix}{q}" for q in queries]
        q_embs = self.model.encode(
            formatted_queries,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Matrix multiplication: (N_passages, D) @ (D, N_queries) -> (N_passages, N_queries)
        scores_matrix = np.dot(self.passage_embeddings, q_embs.T)

        results: list[list[tuple[str, float]]] = []
        top_k = min(top_k, len(self.passage_ids))

        for j in range(len(queries)):
            col_scores = scores_matrix[:, j]
            ranked_indices = np.argsort(-col_scores)[:top_k]
            results.append([(self.passage_ids[i], float(col_scores[i])) for i in ranked_indices])

        return results
