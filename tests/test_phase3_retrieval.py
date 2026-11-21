"""Unit tests for Phase 3 Multi-Strategy Retrieval Pipeline.

Verifies:
1. Retrieval ranks and metric calculations on controlled examples.
2. Gold evidence ID retention throughout retrieval and reranking.
3. Evaluation corpus isolation (Dev vs. Test search corpora).
4. Deterministic reproducibility across repeated runs.
5. Empty and missing evidence handling for unanswerable queries.
6. Multilingual tokenization and RRF fusion formulas.
"""

from __future__ import annotations

import json
import math

from indicrag.retrieval.bm25 import BM25Retriever, tokenize_multilingual
from indicrag.retrieval.evaluator import (
    RetrievalEvaluator,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
)
from indicrag.retrieval.rrf import reciprocal_rank_fusion
from indicrag.retrieval.run_benchmark import build_gold_passage_mapping

# ---------------------------------------------------------------------------
# 1. Controlled Metrics Tests
# ---------------------------------------------------------------------------

def test_metrics_calculation_controlled_examples():
    """Verify Recall@k, MRR@10, and nDCG@10 on exact hand-calculated test cases."""
    gold_set = {"psg_gold_1"}

    # Case A: Gold at rank 1
    ranked_a = ["psg_gold_1", "psg_other_1", "psg_other_2"]
    assert compute_recall_at_k(ranked_a, gold_set, k=1) == 1.0
    assert compute_recall_at_k(ranked_a, gold_set, k=5) == 1.0
    assert compute_mrr_at_k(ranked_a, gold_set, k=10) == 1.0
    assert compute_ndcg_at_k(ranked_a, gold_set, k=10) == 1.0

    # Case B: Gold at rank 2
    ranked_b = ["psg_other_1", "psg_gold_1", "psg_other_2"]
    assert compute_recall_at_k(ranked_b, gold_set, k=1) == 0.0
    assert compute_recall_at_k(ranked_b, gold_set, k=2) == 1.0
    assert compute_mrr_at_k(ranked_b, gold_set, k=10) == 0.5
    # DCG = 1 / log2(3), IDCG = 1 / log2(2) = 1.0 -> nDCG = 1 / log2(3)
    expected_ndcg_b = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert abs(compute_ndcg_at_k(ranked_b, gold_set, k=10) - expected_ndcg_b) < 1e-6

    # Case C: Gold at rank 11 (outside top 10)
    ranked_c = [f"psg_other_{i}" for i in range(10)] + ["psg_gold_1"]
    assert compute_recall_at_k(ranked_c, gold_set, k=10) == 0.0
    assert compute_recall_at_k(ranked_c, gold_set, k=20) == 1.0
    assert compute_mrr_at_k(ranked_c, gold_set, k=10) == 0.0
    assert compute_ndcg_at_k(ranked_c, gold_set, k=10) == 0.0

    # Case D: Gold not in retrieved list
    ranked_d = ["psg_other_1", "psg_other_2"]
    assert compute_recall_at_k(ranked_d, gold_set, k=10) == 0.0
    assert compute_mrr_at_k(ranked_d, gold_set, k=10) == 0.0
    assert compute_ndcg_at_k(ranked_d, gold_set, k=10) == 0.0


# ---------------------------------------------------------------------------
# 2. Unanswerable Query & Empty Evidence Handling
# ---------------------------------------------------------------------------

def test_unanswerable_query_empty_evidence_handling():
    """Verify that unanswerable queries (empty gold set) never produce divide-by-zero."""
    empty_gold: set[str] = set()
    ranked = ["psg_1", "psg_2", "psg_3"]

    assert compute_recall_at_k(ranked, empty_gold, k=5) == 0.0
    assert compute_mrr_at_k(ranked, empty_gold, k=10) == 0.0
    assert compute_ndcg_at_k(ranked, empty_gold, k=10) == 0.0

    # Evaluator runner with unanswerable queries
    evaluator = RetrievalEvaluator(k_values=[1, 5, 10])
    dummy_queries = [
        {"id": "unans_1", "question": "काल्पनिक सवाल?", "is_impossible": True},
        {"id": "unans_2", "question": "दूसरा सवाल?", "is_impossible": True},
    ]
    gold_map = {"unans_1": set(), "unans_2": set()}

    metrics = evaluator.evaluate_retrieval(
        queries=dummy_queries,
        gold_passage_map=gold_map,
        retriever_func=lambda q, top_k: [("psg_1", 1.0), ("psg_2", 0.5)],
        top_k=5,
    )
    assert metrics["num_unanswerable"] == 2
    assert metrics["num_answerable"] == 0
    assert metrics["recall@5"] == 0.0
    assert metrics["mrr@10"] == 0.0
    assert metrics["ndcg@10"] == 0.0


# ---------------------------------------------------------------------------
# 3. Multilingual Tokenization & BM25 Tests
# ---------------------------------------------------------------------------

def test_multilingual_tokenization():
    """Verify tokenization preserves Devanagari words, Latin words, and numbers while stripping punctuation."""
    text = "बिंजेन और बॉन (Bingen and Bonn) के बीच 1980 में?"
    tokens = tokenize_multilingual(text)
    assert "बिंजेन" in tokens
    assert "और" in tokens
    assert "बॉन" in tokens
    assert "bingen" in tokens
    assert "bonn" in tokens
    assert "1980" in tokens
    assert "(" not in tokens
    assert "?" not in tokens


def test_bm25_retrieval_ranking():
    """Verify BM25 index ranks exact matching passage highest."""
    corpus = [
        {"passage_id": "p1", "title": "राइन", "text": "बिंजेन और बॉन के बीच मध्य राइन बहती है।"},
        {"passage_id": "p2", "title": "गंगा", "text": "गंगा नदी हिमालय से निकलती है और बंगाल की खाड़ी में गिरती है।"},
        {"passage_id": "p3", "title": "यमुना", "text": "यमुना नदी दिल्ली और आगरा से होकर गुजरती है।"},
    ]
    retriever = BM25Retriever(corpus, k1=1.5, b=0.75)
    results = retriever.retrieve("राइन नदी बिंजेन बॉन", top_k=3)
    assert len(results) == 3
    # Top passage must be p1
    top_pid, top_score = results[0]
    assert top_pid == "p1"
    assert top_score > 0.0


# ---------------------------------------------------------------------------
# 4. RRF Fusion Mathematical Verification
# ---------------------------------------------------------------------------

def test_reciprocal_rank_fusion_math():
    """Verify RRF formula calculation with c=60."""
    # List 1: p1 (rank 1), p2 (rank 2), p3 (rank 3)
    # List 2: p2 (rank 1), p3 (rank 2), p1 (rank 3)
    list1 = [("p1", 10.0), ("p2", 8.0), ("p3", 5.0)]
    list2 = [("p2", 0.9), ("p3", 0.7), ("p1", 0.4)]

    # Scores with c=60:
    # p1: 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.0163934 + 0.0158730 = 0.0322664
    # p2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.0161290 + 0.0163934 = 0.0325224
    # p3: 1/(60+3) + 1/(60+2) = 1/63 + 1/62 = 0.0158730 + 0.0161290 = 0.0320020
    # Expected ranking: p2 > p1 > p3

    fused = reciprocal_rank_fusion([list1, list2], c=60, top_k=3)
    assert len(fused) == 3
    assert fused[0][0] == "p2"
    assert fused[1][0] == "p1"
    assert fused[2][0] == "p3"

    expected_p2 = (1.0 / 62) + (1.0 / 61)
    assert abs(fused[0][1] - expected_p2) < 1e-6


# ---------------------------------------------------------------------------
# 5. Corpus Isolation & Gold Evidence Retention
# ---------------------------------------------------------------------------

def test_evaluation_corpus_isolation():
    """Verify Dev queries evaluate only on Dev search corpus; Test queries only on Test corpus."""
    with open("data/processed/corpus_dev.json", encoding="utf-8") as f:
        corpus_dev = json.load(f)
    with open("data/processed/corpus_test.json", encoding="utf-8") as f:
        corpus_test = json.load(f)
    with open("data/processed/hindi_dev.json", encoding="utf-8") as f:
        dev_titles = {r["title"] for r in json.load(f)}
    with open("data/processed/hindi_test.json", encoding="utf-8") as f:
        test_titles = {r["title"] for r in json.load(f)}

    # Dev corpus must contain zero Test titles
    dev_corpus_titles = {p["title"] for p in corpus_dev}
    assert len(dev_corpus_titles & test_titles) == 0

    # Test corpus must contain zero Dev titles
    test_corpus_titles = {p["title"] for p in corpus_test}
    assert len(test_corpus_titles & dev_titles) == 0


def test_gold_evidence_presence_and_retention():
    """Verify that every answerable Dev query has its gold passage present and matched."""
    with open("data/processed/benchmark_dev_150.json", encoding="utf-8") as f:
        benchmark_dev = json.load(f)
    with open("data/processed/corpus_dev.json", encoding="utf-8") as f:
        corpus_dev = json.load(f)

    gold_map = build_gold_passage_mapping(benchmark_dev, corpus_dev)

    ans_queries = [q for q in benchmark_dev if not q.get("is_impossible", False)]
    assert len(ans_queries) == 75

    for q in ans_queries:
        qid = str(q["id"])
        gold_pids = gold_map.get(qid, set())
        assert len(gold_pids) > 0, f"Query {qid} missing gold passage ID in corpus_dev"


# ---------------------------------------------------------------------------
# 6. Reproducibility
# ---------------------------------------------------------------------------

def test_deterministic_retrieval_reproducibility():
    """Verify that repeated runs of BM25 on the same queries produce bitwise identical rankings."""
    with open("data/processed/corpus_dev.json", encoding="utf-8") as f:
        corpus_dev = json.load(f)

    bm25 = BM25Retriever(corpus_dev[:50], k1=1.5, b=0.75)
    query = "बिंजेन और बॉन के बीच क्या बहती है?"

    run1 = bm25.retrieve(query, top_k=5)
    run2 = bm25.retrieve(query, top_k=5)

    assert run1 == run2, "BM25 retrieval was non-deterministic across repeated executions"
