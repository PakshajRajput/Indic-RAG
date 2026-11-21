# Phase 3 Walkthrough: Multi-Strategy Retrieval Pipeline

## 1. Executive Summary

Phase 3 implements and evaluates the multi-strategy retrieval ladder for **PJ-09 IndicRAG-QA** across clean and perturbed search corpora for Hindi (Devanagari) and Hinglish (Roman script) QA.

In accordance with **AGENTS.md** and user requirements:
- **Zero test-leakage:** Parameter tuning and comparison are performed strictly on the **Dev** search partition (`data/processed/corpus_dev.json` and `corpus_dev_perturbed.json`) with held-out **Test** data untouched.
- **Independent retrieval strategies evaluated:**
  - **R1:** BM25 Lexical Retrieval with script-aware combining-mark tokenization.
  - **R2:** Dense Multilingual Embedding Retrieval (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) with persistent on-disk embedding vector caching.
  - **R3:** Cross-Encoder Reranker (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) with pair-score caching.
  - **R4:** Hybrid Reciprocal Rank Fusion (RRF, $c=60$) combining BM25 and Dense candidate lists.
- **Resource Discipline:** Respecting the CPU-only, ~2.4 GB RAM environment, all embeddings are cached on disk (`data/processed/embeddings/`), Hugging Face caches are redirected to `D:\hf_cache`, and models are loaded sequentially without memory exhaustion.
- **Test Suite Status:** **36 / 36 automated tests passing (100% green)**.

---

## 2. Experimental Configuration & Provenance

| Parameter | Configuration Value | Justification / Source |
| :--- | :--- | :--- |
| **Random Seed** | `42` | AGENTS.md Rule 5 (deterministic reproducibility) |
| **Dev Clean Corpus** | `data/processed/corpus_dev.json` (2,078 unique passages, SHA-256 `d294086976c71200e004b2976048459b7e22b21f5c89c2e5d937eaebf1596d38`) | Phase 1 baseline (1,845 train + 233 dev passages) |
| **Dev Perturbed Corpus** | `data/processed/counterfactual/corpus_dev_perturbed.json` (2,493 passages, SHA-256 `ab0ee0f18c879897f20de77142e20662b928484777aedf98d35610d7c567491e`) | Phase 2 noise & counterfactual injection (2,078 clean + 415 distractors) |
| **Dev Benchmark Queries** | `data/processed/benchmark_dev_150.json` (150 queries: 75 ans, 75 unans) | Balanced connected-component partition |
| **BM25 Parameters (R1)** | $k_1 = 1.5, b = 0.75$, `include_title = True` | Standard Okapi BM25 with `[\w\u0900-\u097F]+` tokenization |
| **Dense Model (R2)** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118M params, 384-dim, Revision: `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` |
| **Reranker Model (R3)** | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 118M params, Revision: `1427fd652930e4ba29e8149678df786c240d8825` |
| **Hybrid RRF (R4)** | $c = 60$, weights $=[1.0, 1.0]$, candidate depth $=20$, output depth $=10$ | Cormack et al. (2009) reciprocal rank fusion |
| **Embeddings Cache** | `data/processed/embeddings/` | SHA-256 keyed `.npy` (2,078 clean, 2,493 perturbed) |
| **Benchmark Results** | `results/retrieval_benchmark.csv` and `retrieval_benchmark.json` | Reproducible benchmark outputs |

---

## 3. Empirical Results & Metrics

All metrics below are computed directly from `results/retrieval_benchmark.csv` across the 75 answerable queries in `benchmark_dev_150.json`. Unanswerable queries (75 items with empty gold evidence) are tracked separately and verified to avoid division-by-zero errors.

### Table 1: Retrieval Ladder on Devanagari Hindi Queries

| Strategy | Corpus Condition | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Latency (ms/q) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R1: BM25 Lexical** | **Clean** | **42.67%** | 82.67% | 86.67% | **90.67%** | 0.6052 | 0.6704 | **5.58 ms** |
| **R2: Dense MiniLM** | **Clean** | 30.67% | 52.00% | 66.67% | 77.33% | 0.4157 | 0.4748 | 20.33 ms |
| **R4: Hybrid RRF** | **Clean** | 33.33% | **82.67%** | **90.67%** | **90.67%** | 0.5377 | 0.6284 | 25.28 ms |
| **R3: Cross-Encoder** | **Clean** | **42.67%** | **82.67%** | 88.00% | 88.00% | **0.6182** | **0.6839** | 1,744.00 ms |
| | | | | | | | | |
| **R1: BM25 Lexical** | **Perturbed** | **42.67%** | 81.33% | 82.67% | 89.33% | 0.5880 | 0.6479 | **8.46 ms** |
| **R2: Dense MiniLM** | **Perturbed** | 24.00% | 52.00% | 61.33% | 74.67% | 0.3642 | 0.4241 | 20.15 ms |
| **R4: Hybrid RRF** | **Perturbed** | 30.67% | 80.00% | **90.67%** | **90.67%** | 0.5115 | 0.6080 | 26.31 ms |
| **R3: Cross-Encoder** | **Perturbed** | 41.33% | **84.00%** | 88.00% | 88.00% | **0.5991** | **0.6692** | 2,272.41 ms |

---

### Table 2: Cross-Script Evaluation: Roman Hinglish Queries against Devanagari Search Corpus

| Strategy | Corpus Condition | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Latency (ms/q) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **R1: BM25 Lexical** | Clean | 0.00% | 0.00% | 0.00% | 0.0000 | 0.0000 | 3.86 ms |
| **R2: Dense MiniLM** | Clean | 0.00% | 1.33% | 1.33% | 0.0067 | 0.0084 | 19.55 ms |
| **R4: Hybrid RRF** | Clean | 0.00% | 1.33% | 1.33% | 0.0033 | 0.0057 | 26.06 ms |
| **R3: Cross-Encoder** | Clean | 0.00% | 0.00% | 1.33% | 0.0013 | 0.0039 | 986.08 ms |
| | | | | | | | |
| **R1: BM25 Lexical** | Perturbed | 0.00% | 0.00% | 0.00% | 0.0000 | 0.0000 | 13.49 ms |
| **R2: Dense MiniLM** | Perturbed | 0.00% | 1.33% | 1.33% | 0.0067 | 0.0084 | 58.62 ms |
| **R4: Hybrid RRF** | Perturbed | 0.00% | 1.33% | 1.33% | 0.0033 | 0.0057 | 76.16 ms |
| **R3: Cross-Encoder** | Perturbed | 0.00% | 0.00% | 1.33% | 0.0015 | 0.0040 | 2,149.70 ms |

---

## 4. Key Scientific & Architectural Findings

### Finding 1: Complementary Strengths of BM25 and Hybrid RRF
- On the clean Hindi corpus, **BM25 (R1)** achieves high initial precision (MRR@10 = 0.6052, Recall@10 = 86.67%).
- Combining BM25 with Dense multilingual retrieval via **Reciprocal Rank Fusion (R4)** achieves the highest overall recall: **90.67% Recall@10** (68 out of 75 answerable queries find their gold context in top 10).
- Crucially, under the **Perturbed Corpus** (injected with 415 counterfactual distractors and topically similar negatives), BM25 Recall@10 drops from 86.67% to 82.67%, and Dense drops from 66.67% to 61.33%.
- **RRF Robustness Finding:** **Hybrid RRF preserves Recall@10 at 90.67%** under the specific 20% distractors tested. We do not claim universal immunity to all forms of noise, but demonstrate that fusing rank signals provides superior noise resilience compared to single-retriever strategies.

### Finding 2: Cross-Encoder Precision Uplift vs. Latency Tradeoff
- The **Cross-Encoder Reranker (R3)** re-sorts top-20 candidates using full cross-attention, boosting MRR@10 from 0.5377 (RRF) to **0.6182** on clean corpus and from 0.5115 (RRF) to **0.5991** on perturbed corpus.
- Similarly, nDCG@10 increases to **0.6839** (clean) and **0.6692** (perturbed).
- **Latency cost:** On a CPU-only host, cross-encoder scoring of top-20 candidates requires ~1,700–2,200 ms per query compared to ~5 ms for BM25 and ~25 ms for RRF. In serving pipelines, RRF provides optimal latency/recall efficiency, while Cross-Encoder provides maximum precision.

### Finding 3: Preserving the Cross-Script Research Question
- When Roman-script Hinglish queries are submitted directly against a Devanagari corpus without normalization:
  - BM25 lexical recall is **0.00%** because Latin character tokens have zero surface overlap with Devanagari Unicode.
  - Multilingual dense embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) achieve only **1.33% Recall@10**, demonstrating severe script transfer degradation.
- **Scientific Stance:** These results serve as the **unnormalized baseline**. We treat query normalization not as an assumed necessity, but as a formal hypothesis to be ablated in Phase 6. A controlled comparison will evaluate:
  1. Unnormalized Roman Hinglish (Baseline)
  2. Rule-based phonetic transliteration (IndicXlit / deterministic rules)
  3. LLM contextual normalization
  against identical queries and corpus versions.

---

## 5. Automated Verification Test Suite

Test suite execution (`pytest tests/ -v`):
```text
============================= test session starts =============================
platform win32 -- Python 3.10.6, pytest-9.1.1, pluggy-1.7.0
rootdir: D:\Indic-QA-RAG, configfile: pyproject.toml
collected 36 items

tests/test_hinglish_pipeline.py::test_validator_clean_roman PASSED       [  2%]
tests/test_hinglish_pipeline.py::test_validator_devanagari_leakage PASSED [  5%]
tests/test_hinglish_pipeline.py::test_validator_missing_digits PASSED    [  8%]
tests/test_hinglish_pipeline.py::test_validator_duplicate_detection PASSED [ 11%]
tests/test_hinglish_pipeline.py::test_disk_cache_roundtrip PASSED        [ 13%]
tests/test_hinglish_pipeline.py::test_generator_with_stub_client PASSED  [ 16%]
tests/test_leakage.py::test_zero_title_leakage PASSED                    [ 19%]
tests/test_leakage.py::test_zero_normalized_title_leakage PASSED         [ 22%]
tests/test_leakage.py::test_zero_context_leakage PASSED                  [ 25%]
tests/test_leakage.py::test_zero_question_leakage PASSED                 [ 27%]
tests/test_leakage.py::test_zero_id_leakage PASSED                       [ 30%]
tests/test_leakage.py::test_benchmark_subset_balance PASSED              [ 33%]
tests/test_leakage.py::test_record_schema_and_hinglish_link PASSED       [ 36%]
tests/test_leakage.py::test_connected_component_clustering PASSED        [ 38%]
tests/test_leakage.py::test_gold_evidence_present_in_search_corpus PASSED [ 41%]
tests/test_leakage.py::test_corpus_isolation_and_no_annotation_leakage PASSED [ 44%]
tests/test_leakage.py::test_qualified_leakage_assertion_scope PASSED     [ 47%]
tests/test_phase0_smoke.py::test_pydantic_state_schema PASSED            [ 50%]
tests/test_phase0_smoke.py::test_langgraph_stub_graph PASSED             [ 52%]
tests/test_phase0_smoke.py::test_budget_enforcement PASSED               [ 55%]
tests/test_phase0_smoke.py::test_no_leakage_stub PASSED                  [ 58%]
tests/test_phase0_smoke.py::test_reason_codes PASSED                     [ 61%]
tests/test_phase2_noise.py::test_baseline_manifest_preservation PASSED   [ 63%]
tests/test_phase2_noise.py::test_deterministic_reproduction PASSED       [ 66%]
tests/test_phase2_noise.py::test_rgb_noise_bounds_and_structure PASSED   [ 69%]
tests/test_phase2_noise.py::test_zero_annotation_leakage_in_perturbed_corpora PASSED [ 72%]
tests/test_phase2_noise.py::test_counterfactual_distractor_provenance PASSED [ 75%]
tests/test_phase2_noise.py::test_distractor_does_not_inject_gold_answer PASSED [ 77%]
tests/test_phase3_retrieval.py::test_metrics_calculation_controlled_examples PASSED [ 80%]
tests/test_phase3_retrieval.py::test_unanswerable_query_empty_evidence_handling PASSED [ 83%]
tests/test_phase3_retrieval.py::test_multilingual_tokenization PASSED    [ 86%]
tests/test_phase3_retrieval.py::test_bm25_retrieval_ranking PASSED       [ 88%]
tests/test_phase3_retrieval.py::test_reciprocal_rank_fusion_math PASSED  [ 91%]
tests/test_phase3_retrieval.py::test_evaluation_corpus_isolation PASSED  [ 94%]
tests/test_phase3_retrieval.py::test_gold_evidence_presence_and_retention PASSED [ 97%]
tests/test_phase3_retrieval.py::test_deterministic_retrieval_reproducibility PASSED [100%]

============================= 36 passed in 10.61s =============================
```

---

## 6. Known Limitations & Phase 3 Gate Status

1. **Hardware & Latency Constraints:** On the CPU-only host with ~2.4 GB free RAM, R3 (Cross-Encoder Reranker) is CPU-intensive (~1.7–2.2s per query). While feasible for batch evaluation, online serving should rely primarily on R4 (Hybrid RRF) or stage reranking strictly on top-5 candidates.
2. **Hinglish Cross-Script Gap:** Native retrieval models cannot bridge the script gap directly. The Query Agent must normalize Roman-script queries to Devanagari before corpus lookup.
3. **Phase Gate:** In accordance with user rules and AGENTS.md Rule 4, execution is halted at the Phase 3 gate pending user review and approval before beginning Phase 4.
