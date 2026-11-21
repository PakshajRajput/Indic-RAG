"""End-to-End Retrieval Benchmark Runner for PJ-09 IndicRAG-QA (Phase 3).

Executes the complete retrieval ladder:
- R1: BM25 Lexical Retrieval
- R2: Dense Multilingual Retrieval
- R3: Cross-Encoder Reranking
- R4: Hybrid Reciprocal Rank Fusion (RRF)

Evaluates on Dev Benchmark across:
- Clean Dev Corpus vs. Perturbed Dev Corpus
- Devanagari Hindi vs. Roman-script Hinglish queries
Saves full metrics and comparison tables to results/
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import yaml

from indicrag.hinglish.transliterate import hindi_to_hinglish
from indicrag.retrieval.bm25 import BM25Retriever
from indicrag.retrieval.dense import DenseRetriever
from indicrag.retrieval.evaluator import RetrievalEvaluator
from indicrag.retrieval.reranker import CrossEncoderReranker
from indicrag.retrieval.rrf import HybridRRFRetriever

# Ensure HF cache is strictly on D: drive
os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def build_gold_passage_mapping(
    queries: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """Build mapping from query ID to set of matching gold passage IDs."""
    text_to_pids: dict[str, set[str]] = {}
    for p in corpus:
        norm_text = p.get("text", "").strip()
        pid = str(p.get("passage_id", p.get("id", "")))
        text_to_pids.setdefault(norm_text, set()).add(pid)

    gold_map: dict[str, set[str]] = {}
    for q in queries:
        qid = str(q["id"])
        if q.get("is_impossible", False):
            gold_map[qid] = set()
        else:
            ctx = q.get("context", "").strip()
            matched = text_to_pids.get(ctx, set())
            gold_map[qid] = matched
    return gold_map


def build_passage_text_lookup(corpus: list[dict[str, Any]]) -> dict[str, str]:
    """Build fast lookup mapping passage_id to 'title text'."""
    lookup: dict[str, str] = {}
    for p in corpus:
        pid = str(p.get("passage_id", p.get("id", "")))
        title = p.get("title", "")
        text = p.get("text", "")
        lookup[pid] = f"{title} {text}" if title else text
    return lookup


def prepare_query_sets(
    benchmark_queries: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Prepare Hindi and Roman Hinglish query collections."""
    hindi_queries: list[dict[str, Any]] = []
    hinglish_queries: list[dict[str, Any]] = []

    for q in benchmark_queries:
        hindi_q = {
            "id": q["id"],
            "question": q["question"],
            "is_impossible": q.get("is_impossible", False),
            "script": "hindi",
        }
        hindi_queries.append(hindi_q)

        # Transliterate to Roman-script Hinglish
        hinglish_text = hindi_to_hinglish(q["question"])
        hinglish_q = {
            "id": q["id"],
            "question": hinglish_text,
            "is_impossible": q.get("is_impossible", False),
            "script": "hinglish",
            "synthetic_hinglish_id": q.get("synthetic_hinglish_id", f"{q['id']}_hinglish"),
        }
        hinglish_queries.append(hinglish_q)

    return {
        "hindi": hindi_queries,
        "hinglish": hinglish_queries,
    }


def run_benchmark(
    config_path: str = "configs/retrieval_config.yaml",
    dense_model_name: str | None = None,
    eval_test: bool = False,
) -> dict[str, Any]:
    """Run complete retrieval benchmark over Dev (and optionally Test) data."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load corpus files
    corpus_dev_path = cfg["paths"]["corpus_dev"]
    corpus_dev_pert_path = cfg["paths"]["corpus_dev_perturbed"]
    benchmark_dev_path = cfg["paths"]["benchmark_dev"]

    with open(corpus_dev_path, encoding="utf-8") as f:
        corpus_dev_clean = json.load(f)
    with open(corpus_dev_pert_path, encoding="utf-8") as f:
        corpus_dev_perturbed = json.load(f)
    with open(benchmark_dev_path, encoding="utf-8") as f:
        benchmark_dev_queries = json.load(f)

    query_sets = prepare_query_sets(benchmark_dev_queries)

    gold_map_clean = build_gold_passage_mapping(benchmark_dev_queries, corpus_dev_clean)
    gold_map_perturbed = build_gold_passage_mapping(benchmark_dev_queries, corpus_dev_perturbed)

    lookup_clean = build_passage_text_lookup(corpus_dev_clean)
    lookup_perturbed = build_passage_text_lookup(corpus_dev_perturbed)

    evaluator = RetrievalEvaluator(k_values=cfg["evaluation"]["k_values"])

    # Configure models
    dense_model = dense_model_name or cfg["r2_dense"]["model_name"]
    reranker_model = cfg["r3_reranker"]["model_name"]

    experiments: list[dict[str, Any]] = []

    corpora_eval = [
        ("clean", corpus_dev_clean, gold_map_clean, lookup_clean),
        ("perturbed", corpus_dev_perturbed, gold_map_perturbed, lookup_perturbed),
    ]

    for corpus_condition, corpus, gold_map, passage_lookup in corpora_eval:
        print("\n========================================================")
        print(f"Indexing Corpus: Dev ({corpus_condition.upper()}) | Passages: {len(corpus)}")
        print("========================================================")

        # 1. Initialize R1: BM25
        print("Building BM25 index...")
        bm25 = BM25Retriever(
            corpus,
            k1=cfg["r1_bm25"]["k1"],
            b=cfg["r1_bm25"]["b"],
            include_title=cfg["r1_bm25"]["include_title"],
        )

        # 2. Initialize R2: Dense
        print(f"Initializing Dense Retriever ({dense_model})...")
        dense = DenseRetriever(
            corpus,
            model_name_or_path=dense_model,
            cache_dir=cfg["paths"]["embeddings_cache_dir"],
            corpus_id=f"dev_{corpus_condition}",
            batch_size=cfg["r2_dense"]["batch_size"],
            device=cfg["r2_dense"]["device"],
            include_title=cfg["r2_dense"]["include_title"],
        )

        # 3. Initialize R4: Hybrid RRF
        hybrid_rrf = HybridRRFRetriever(
            retrievers=[bm25, dense],
            weights=cfg["r4_rrf"]["weights"],
            c=cfg["r4_rrf"]["c"],
            candidate_k=cfg["r4_rrf"]["candidate_k"],
        )

        # 4. Initialize R3: Reranker
        print(f"Initializing CrossEncoder Reranker ({reranker_model})...")
        reranker = CrossEncoderReranker(
            model_name_or_path=reranker_model,
            batch_size=cfg["r3_reranker"]["batch_size"],
            device=cfg["r3_reranker"]["device"],
        )

        for script_name in ["hindi", "hinglish"]:
            queries = query_sets[script_name]
            print(f"\n--- Evaluating Queries: {script_name.upper()} ({len(queries)} questions) on {corpus_condition.upper()} corpus ---")

            # Strategy R1: BM25
            print("Running R1: BM25...")
            m_bm25 = evaluator.evaluate_retrieval(
                queries=queries,
                gold_passage_map=gold_map,
                retriever_func=lambda q, top_k: bm25.retrieve(q, top_k=top_k),
                top_k=cfg["r1_bm25"]["top_k"],
            )
            exp_bm25 = {
                "strategy": "R1_BM25",
                "corpus_condition": corpus_condition,
                "query_script": script_name,
                "model": "rank_bm25_okapi",
                **m_bm25,
            }
            experiments.append(exp_bm25)
            print(f"  R1_BM25: R@5={m_bm25['recall@5']:.4f}, R@10={m_bm25['recall@10']:.4f}, MRR@10={m_bm25['mrr@10']:.4f}, nDCG@10={m_bm25['ndcg@10']:.4f}, Latency={m_bm25['mean_latency_ms']:.2f}ms")

            # Strategy R2: Dense
            print(f"Running R2: Dense ({dense_model})...")
            m_dense = evaluator.evaluate_retrieval(
                queries=queries,
                gold_passage_map=gold_map,
                retriever_func=lambda q, top_k: dense.retrieve(q, top_k=top_k),
                top_k=cfg["r2_dense"]["top_k"],
            )
            exp_dense = {
                "strategy": "R2_Dense",
                "corpus_condition": corpus_condition,
                "query_script": script_name,
                "model": dense_model,
                **m_dense,
            }
            experiments.append(exp_dense)
            print(f"  R2_Dense: R@5={m_dense['recall@5']:.4f}, R@10={m_dense['recall@10']:.4f}, MRR@10={m_dense['mrr@10']:.4f}, nDCG@10={m_dense['ndcg@10']:.4f}, Latency={m_dense['mean_latency_ms']:.2f}ms")

            # Strategy R4: Hybrid RRF (BM25 + Dense)
            print("Running R4: Hybrid RRF...")
            m_rrf = evaluator.evaluate_retrieval(
                queries=queries,
                gold_passage_map=gold_map,
                retriever_func=lambda q, top_k: hybrid_rrf.retrieve(q, top_k=top_k),
                top_k=cfg["r4_rrf"]["top_k"],
            )
            exp_rrf = {
                "strategy": "R4_Hybrid_RRF",
                "corpus_condition": corpus_condition,
                "query_script": script_name,
                "model": f"BM25 + {dense_model} (c={cfg['r4_rrf']['c']})",
                **m_rrf,
            }
            experiments.append(exp_rrf)
            print(f"  R4_RRF:   R@5={m_rrf['recall@5']:.4f}, R@10={m_rrf['recall@10']:.4f}, MRR@10={m_rrf['mrr@10']:.4f}, nDCG@10={m_rrf['ndcg@10']:.4f}, Latency={m_rrf['mean_latency_ms']:.2f}ms")

            # Strategy R3: Cross-Encoder Reranker (over R4 Hybrid candidates)
            print("Running R3: Cross-Encoder Reranker over R4 candidates...")
            def rerank_func(q: str, top_k: int) -> list[tuple[str, float]]:
                cands = hybrid_rrf.retrieve(q, top_k=cfg["r3_reranker"]["candidate_k"])
                return reranker.rerank(q, cands, passage_lookup, top_k=top_k)

            m_rerank = evaluator.evaluate_retrieval(
                queries=queries,
                gold_passage_map=gold_map,
                retriever_func=rerank_func,
                top_k=cfg["r3_reranker"]["rerank_top_k"],
            )
            exp_rerank = {
                "strategy": "R3_Reranker",
                "corpus_condition": corpus_condition,
                "query_script": script_name,
                "model": reranker_model,
                **m_rerank,
            }
            experiments.append(exp_rerank)
            print(f"  R3_Rerank: R@5={m_rerank['recall@5']:.4f}, R@10={m_rerank['recall@10']:.4f}, MRR@10={m_rerank['mrr@10']:.4f}, nDCG@10={m_rerank['ndcg@10']:.4f}, Latency={m_rerank['mean_latency_ms']:.2f}ms")

    # Save output artifacts
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / "retrieval_benchmark.json"
    csv_path = results_dir / "retrieval_benchmark.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"experiments": experiments, "config": cfg}, f, indent=2, ensure_ascii=False)

    if experiments:
        keys = list(experiments[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(experiments)

    print("\nBenchmark completed successfully! Results written to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")

    return {"experiments": experiments, "json_path": str(json_path), "csv_path": str(csv_path)}


if __name__ == "__main__":
    run_benchmark()
