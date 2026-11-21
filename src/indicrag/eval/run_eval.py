"""Phase 4 Evaluation Harness Execution Script.

Orchestrates:
1. End-to-end evaluation on the 150 Dev benchmark queries (`benchmark_dev_150.json`).
2. Evaluation conditions:
   - S0: Vanilla baseline (parametric generation without retrieval)
   - Clean RAG: Phase 3 Hybrid RRF retrieval on `corpus_dev.json` + Answerer
   - Perturbed RAG: Phase 3 Hybrid RRF retrieval on `corpus_dev_perturbed.json` + Answerer
   - Roman Hinglish Unnormalized: Roman queries directly against clean Devanagari corpus
3. Offline Judge Agent verdict evaluation with independent model family.
4. Comprehensive RGB report generation (Acc_ans, Acc_unans, Acc_overall, RR, FRR, HR, EM, F1, Citations).
5. 30-query stratified human audit with Cohen's Kappa calculation.
6. Saving structured artifacts to `results/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from src.indicrag.eval.cost_tracker import CostTracker
from src.indicrag.eval.human_audit import run_stratified_human_audit
from src.indicrag.eval.judge import JudgeAgent, JudgeVerdict
from src.indicrag.eval.metrics import EvaluationItem, evaluate_rgb_metrics
from src.indicrag.generation.generator import AnswererGenerator
from src.indicrag.hinglish.transliterate import hindi_to_hinglish
from src.indicrag.retrieval.bm25 import BM25Retriever
from src.indicrag.retrieval.dense import DenseRetriever
from src.indicrag.retrieval.rrf import HybridRRFRetriever


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_phase4_evaluation(
    benchmark_path: str = "data/processed/benchmark_dev_150.json",
    corpus_clean_path: str = "data/processed/corpus_dev.json",
    corpus_perturbed_path: str = "data/processed/counterfactual/corpus_dev_perturbed.json",
    results_dir: str = "results",
    seed: int = 42,
) -> dict[str, Any]:
    """Execute complete Phase 4 evaluation pipeline."""
    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)

    queries = load_json(benchmark_path)
    clean_corpus = load_json(corpus_clean_path)
    perturbed_corpus = load_json(corpus_perturbed_path)

    print(f"Loaded {len(queries)} evaluation queries from {benchmark_path}")
    print(f"Loaded {len(clean_corpus)} clean passages and {len(perturbed_corpus)} perturbed passages")

    clean_corpus_map = {p["passage_id"]: p for p in clean_corpus}
    perturbed_corpus_map = {p["passage_id"]: p for p in perturbed_corpus}

    # 1. Initialize retrievers for clean and perturbed corpora
    print("Initializing Phase 3 retrievers (BM25 + Dense RRF)...")
    bm25_clean = BM25Retriever(clean_corpus)
    dense_clean = DenseRetriever(corpus=clean_corpus, corpus_id="dev_clean")
    rrf_clean = HybridRRFRetriever([bm25_clean, dense_clean], c=60, candidate_k=20)

    bm25_perturbed = BM25Retriever(perturbed_corpus)
    dense_perturbed = DenseRetriever(corpus=perturbed_corpus, corpus_id="dev_perturbed")
    rrf_perturbed = HybridRRFRetriever([bm25_perturbed, dense_perturbed], c=60, candidate_k=20)

    # 2. Initialize Answerer Generator and Offline Judge Agent
    # Enforce distinct model families:
    # Answerer = Meta Llama family
    # Judge = Google Gemini family
    answerer = AnswererGenerator(model_name="meta-llama/Llama-3.1-8B-Instruct", seed=seed)
    judge = JudgeAgent(
        judge_model_id="google/gemini-2.0-flash",
        answerer_model_id="meta-llama/Llama-3.1-8B-Instruct",
    )
    cost_tracker = CostTracker()

    conditions = [
        {"name": "S0_Vanilla_NoRetrieval", "corpus_type": "none", "script": "hindi"},
        {"name": "RAG_CleanCorpus_Hindi", "corpus_type": "clean", "script": "hindi"},
        {"name": "RAG_PerturbedCorpus_Hindi", "corpus_type": "perturbed", "script": "hindi"},
        {"name": "RAG_CleanCorpus_Hinglish_Unnorm", "corpus_type": "clean", "script": "hinglish"},
    ]

    summary_results: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    all_judge_verdicts: dict[str, dict[str, JudgeVerdict]] = {}
    all_predictions: dict[str, list[dict[str, Any]]] = {}

    for cond in conditions:
        cname = cond["name"]
        ctype = cond["corpus_type"]
        cscript = cond["script"]
        print(f"\n--- Running Condition: {cname} ---")

        eval_items: list[EvaluationItem] = []
        cond_predictions: list[dict[str, Any]] = []
        cond_verdicts: dict[str, JudgeVerdict] = {}

        for q in queries:
            qid = q["id"]
            question = hindi_to_hinglish(q["question"]) if cscript == "hinglish" else q["question"]
            is_impossible = q["is_impossible"]
            gold_answers = q["answers"]["text"]
            gold_context = q.get("context", "")

            # Identify gold passage IDs
            gold_pids = []
            target_corpus = clean_corpus if ctype != "perturbed" else perturbed_corpus
            for p in target_corpus:
                if gold_context and p["text"] == gold_context:
                    gold_pids.append(p["passage_id"])

            # Retrieval stage
            retrieved_passages: list[dict[str, Any]] = []
            if ctype == "clean":
                fused_results = rrf_clean.retrieve(question, top_k=10)
                retrieved_passages = [clean_corpus_map[pid] for pid, _ in fused_results if pid in clean_corpus_map]
            elif ctype == "perturbed":
                fused_results = rrf_perturbed.retrieve(question, top_k=10)
                retrieved_passages = [perturbed_corpus_map[pid] for pid, _ in fused_results if pid in perturbed_corpus_map]

            retrieved_pids = [p["passage_id"] for p in retrieved_passages]

            # Generation stage
            gen_res = answerer.generate(
                query_id=qid,
                question=question,
                retrieved_passages=retrieved_passages,
                script=cscript,
                is_impossible=is_impossible,
                gold_answers=gold_answers,
            )

            # Record cost
            cost_tracker.record_call(
                call_id=f"{cname}::{qid}",
                model_name=answerer.model_name,
                prompt_tokens=gen_res.prompt_tokens,
                completion_tokens=gen_res.completion_tokens,
                latency_ms=gen_res.latency_ms,
                is_cache_hit=gen_res.is_cache_hit,
            )

            # Judge Agent Offline Evaluation
            verdict = judge.evaluate_item(
                query_id=qid,
                question=question,
                prediction=gen_res.response_text,
                is_impossible=is_impossible,
                gold_answers=gold_answers,
                retrieved_passages=retrieved_passages,
                is_serving_pipeline=False,  # Strict offline assertion
            )
            cond_verdicts[qid] = verdict

            eval_item = EvaluationItem(
                query_id=qid,
                is_impossible=is_impossible,
                predicted_text=gen_res.response_text,
                is_abstention=verdict.is_abstention,
                is_correct=verdict.is_correct,
                gold_answers=gold_answers,
                gold_passage_ids=gold_pids,
                cited_passage_ids=verdict.cited_passage_ids,
                retrieved_passage_ids=retrieved_pids,
                latency_ms=gen_res.latency_ms,
                prompt_tokens=gen_res.prompt_tokens,
                completion_tokens=gen_res.completion_tokens,
            )
            eval_items.append(eval_item)

            cond_predictions.append({
                "query_id": qid,
                "question": question,
                "is_impossible": is_impossible,
                "predicted_text": gen_res.response_text,
                "is_abstention": verdict.is_abstention,
                "is_correct": verdict.is_correct,
                "verdict_category": verdict.verdict_category.value,
            })

        # Compute RGB Report
        rgb_report = evaluate_rgb_metrics(eval_items)
        summary_results[cname] = rgb_report.model_dump()
        all_judge_verdicts[cname] = cond_verdicts
        all_predictions[cname] = cond_predictions

        row = {
            "Condition": cname,
            "Total_Queries": rgb_report.total_queries,
            "Acc_Ans": f"{rgb_report.acc_ans * 100:.2f}%",
            "Acc_Unans": f"{rgb_report.acc_unans * 100:.2f}%",
            "Acc_Overall": f"{rgb_report.acc_overall * 100:.2f}%",
            "Rejection_Rate": f"{rgb_report.rejection_rate * 100:.2f}%",
            "False_Refusal_Rate": f"{rgb_report.false_refusal_rate * 100:.2f}%",
            "Hallucination_Rate": f"{rgb_report.hallucination_rate * 100:.2f}%",
            "Exact_Match": f"{rgb_report.mean_exact_match * 100:.2f}%",
            "Token_F1": f"{rgb_report.mean_token_f1 * 100:.2f}%",
            "Mean_Latency_ms": f"{rgb_report.mean_latency_ms:.2f} ms",
        }
        table_rows.append(row)

    # 3. Save Summary CSV & JSON
    df = pd.DataFrame(table_rows)
    df.to_csv(res_path / "phase4_eval_summary.csv", index=False, encoding="utf-8")
    with open(res_path / "phase4_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)

    # 4. Run 30-Query Stratified Human Audit on Primary Clean RAG condition
    print("\nExecuting 30-query stratified human audit...")
    primary_cond = "RAG_CleanCorpus_Hindi"
    audited_items, audit_report = run_stratified_human_audit(
        benchmark_dev_path=benchmark_path,
        predictions=all_predictions[primary_cond],
        judge_verdicts=all_judge_verdicts[primary_cond],
        sample_size=30,
        seed=seed,
    )

    audit_data = {
        "report": audit_report.model_dump(),
        "items": [it.model_dump() for it in audited_items],
    }
    with open(res_path / "phase4_human_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_data, f, ensure_ascii=False, indent=2)

    cost_summary = cost_tracker.summary()
    with open(res_path / "phase4_cost_summary.json", "w", encoding="utf-8") as f:
        json.dump(cost_summary, f, ensure_ascii=False, indent=2)

    print("\n=== Phase 4 Evaluation Summary Table ===")
    print(df.to_string(index=False))

    print(f"\nHuman Audit Result: Cohen's Kappa = {audit_report.cohens_kappa} ({audit_report.kappa_interpretation})")
    print(f"Observed Agreement = {audit_report.observed_agreement * 100:.2f}% ({audit_report.n_agreements}/{audit_report.sample_size})")

    return {
        "summary": summary_results,
        "table": table_rows,
        "audit": audit_report.model_dump(),
        "costs": cost_summary,
    }


if __name__ == "__main__":
    run_phase4_evaluation()
