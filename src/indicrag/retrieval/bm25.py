"""BM25 Lexical Retriever (R1) for Hindi, Hinglish, and English text.

Uses rank_bm25 with script-aware tokenization and configurable k1, b parameters.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize_multilingual(text: str) -> list[str]:
    """Tokenize Hindi (Devanagari), Hinglish (Roman), and English text.
    
    Normalizes unicode, lowercases Latin characters, strips punctuation,
    and returns non-empty word tokens.
    """
    if not text:
        return []
    # Unicode normalization to NFC
    norm_text = unicodedata.normalize("NFC", text)
    # Lowercase Latin characters
    norm_text = norm_text.lower()
    # Match alphanumeric Latin/digits and all Devanagari script characters (including matras and combining marks)
    tokens = re.findall(r"[\w\u0900-\u097F]+", norm_text)
    return tokens


class BM25Retriever:
    """BM25 Lexical Retriever (R1) over a searchable corpus."""

    def __init__(
        self,
        corpus: Sequence[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
        include_title: bool = True,
    ) -> None:
        """Initialize BM25 index over the provided corpus.
        
        Args:
            corpus: List of passage dictionaries, each with 'passage_id' and 'text'.
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
            include_title: Whether to prepend the passage 'title' to the text.
        """
        self.k1 = k1
        self.b = b
        self.include_title = include_title
        self.corpus = corpus
        self.passage_ids: list[str] = []
        self.corpus_passages: list[str] = []
        self.tokenized_corpus: list[list[str]] = []

        self._build_index()

    def _build_index(self) -> None:
        """Tokenize corpus and build BM25Okapi index."""
        self.passage_ids = []
        self.corpus_passages = []
        self.tokenized_corpus = []

        for item in self.corpus:
            pid = str(item.get("passage_id", item.get("id", "")))
            title = item.get("title", "")
            text = item.get("text", item.get("context", ""))

            if self.include_title and title:
                full_text = f"{title} {text}"
            else:
                full_text = text

            tokens = tokenize_multilingual(full_text)
            self.passage_ids.append(pid)
            self.corpus_passages.append(full_text)
            self.tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Retrieve top_k passages for a query.
        
        Args:
            query: Input query string.
            top_k: Number of candidate passages to return.
            
        Returns:
            List of (passage_id, score) tuples sorted in descending order of score.
        """
        query_tokens = tokenize_multilingual(query)
        if not query_tokens or len(self.passage_ids) == 0:
            # Return top_k arbitrary or empty if corpus is empty
            return [(pid, 0.0) for pid in self.passage_ids[:top_k]]

        doc_scores = self.bm25.get_scores(query_tokens)

        # Rank document indices by score
        # Using top-k partition or argsort
        top_k = min(top_k, len(self.passage_ids))
        if top_k <= 0:
            return []

        # Sort indices by score descending
        ranked_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        return [(self.passage_ids[i], float(doc_scores[i])) for i in ranked_indices]

    def batch_retrieve(
        self, queries: Sequence[str], top_k: int = 10
    ) -> list[list[tuple[str, float]]]:
        """Retrieve top_k passages for a batch of queries."""
        return [self.retrieve(q, top_k=top_k) for q in queries]
