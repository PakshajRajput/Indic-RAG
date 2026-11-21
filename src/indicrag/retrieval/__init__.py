"""Retrieval module for PJ-09 IndicRAG-QA.

Implements multi-strategy retrieval ladder:
- R1: Lexical retrieval (BM25)
- R2: Multilingual dense retrieval (SentenceTransformer)
- R3: Cross-encoder reranker
- R4: Hybrid reciprocal rank fusion (RRF)
"""

from indicrag.retrieval.bm25 import BM25Retriever
from indicrag.retrieval.dense import DenseRetriever
from indicrag.retrieval.evaluator import RetrievalEvaluator
from indicrag.retrieval.reranker import CrossEncoderReranker
from indicrag.retrieval.rrf import reciprocal_rank_fusion

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "CrossEncoderReranker",
    "reciprocal_rank_fusion",
    "RetrievalEvaluator",
]
