# Phase 2 Walkthrough: Corpus Preparation, Noise Injection, and Counterfactual Distractors

## 1. What Was Executed in Phase 2

1. **Baseline Manifest Preservation (Requirement 1):**
   - Cryptographically hashed all Phase 1 clean artifacts into `data/processed/baseline_manifest.json` using SHA-256.
   - Automated test `test_baseline_manifest_preservation` verifies that clean corpora and benchmark files remain 100% byte-identical throughout Phase 2.

2. **Controlled Noise Injection per RGB Protocol (Requirement 2 & Research Detail):**
   - Implemented `NoiseInjector` in `src/indicrag/data/noise.py` reproducing the exact RGB evaluation protocol (Chen et al., AAAI 2024):
     - Passage count: $k=5$.
     - Noise rates: $r \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$.
     - Positive/negative allocation: $\text{neg\_num} = \text{round}(k \cdot r)$, $\text{pos\_num} = k - \text{neg\_num}$.
   - **Answerable Queries:**
     - Under $r < 1.0$: candidate set contains gold evidence + $k - \text{pos\_num}$ noise passages.
     - Under $r = 1.0$: candidate set contains 5 noise passages (0 gold evidence), testing **Negative Rejection**.
   - **Unanswerable Queries:**
     - Across all $r \in [0.0, 1.0]$: candidate set contains 5 distractor/noise passages (0 gold evidence), testing **Abstention Consistency**.

3. **Distinguishing Retrieval Robustness from Answerer Robustness (Research Detail):**
   - **Testbed A (Answerer Robustness / RGB Protocol):**
     - Queries are evaluated against fixed, controlled candidate sets (`data/processed/noise_eval/rgb_dev_eval_sets.json` and `rgb_test_eval_sets.json`).
     - Tests the answerer's resilience to noisy context when gold evidence is guaranteed to be present ($r < 1.0$) vs. absent ($r = 1.0$).
   - **Testbed B (Retriever Robustness / Search Corpus Retrieval):**
     - Queries are evaluated against searchable corpora containing distractors and counterfactual perturbed passages (`data/processed/counterfactual/corpus_dev_perturbed.json` and `corpus_test_perturbed.json`).
     - Tests the retriever's ability to rank the gold passage into top-$k$ in the presence of plausible distractors.

4. **Counterfactual Distractor Generation & Provenance (Requirements 3, 4, 5):**
   - Implemented `CounterfactualGenerator` in `src/indicrag/data/noise.py`:
     - Systematically perturbs years, numbers, and quantities in context paragraphs to create plausible, conflicting distractor passages.
     - Respects forbidden answer constraints: distractors never introduce the gold answer string for unanswerable queries.
   - Provenance Tracking: Every counterfactual passage records `passage_id`, `source_passage_id`, original passage link, modification details, and seed (42).
   - Zero Annotation Leakage: Search corpus passages contain only unannotated text chunks; zero questions, answers, or `is_impossible` labels exist in the searchable corpora.

---

## 2. Test Suite Execution & Metrics

Executed via `D:\Indic-QA-RAG\venv\Scripts\python.exe -m pytest tests/ -v`:

| Test Module | Tests | Status | Execution Time |
|---|---|---|---|
| `tests/test_phase2_noise.py` | 6 | **PASSED** ✅ | 1.15s |
| `tests/test_leakage.py` | 11 | **PASSED** ✅ | 1.19s |
| `tests/test_hinglish_pipeline.py` | 6 | **PASSED** ✅ | 0.12s |
| `tests/test_phase0_smoke.py` | 5 | **PASSED** ✅ | 0.45s |
| **Total Test Suite** | **28 / 28** | **PASSED** (100%) | **2.93s** |

---

## 3. Phase 2 Artifact Inventory

| Artifact Path | Format | Records / Count | Purpose |
|---|---|---|---|
| `data/processed/baseline_manifest.json` | JSON | 9 file hashes | Cryptographic baseline preservation |
| `data/processed/noise_eval/rgb_dev_eval_sets.json` | JSON | 900 candidate sets (150 $\times$ 6 rates) | RGB answerer robustness testbed for Dev |
| `data/processed/noise_eval/rgb_test_eval_sets.json` | JSON | 450 candidate sets (75 $\times$ 6 rates) | RGB answerer robustness testbed for Test |
| `data/processed/counterfactual/corpus_dev_perturbed.json` | JSON | 2,566 passages (2,139 clean + 427 CF) | Searchable Dev corpus with distractors |
| `data/processed/counterfactual/corpus_test_perturbed.json` | JSON | 2,611 passages (2,176 clean + 435 CF) | Searchable Test corpus with distractors |
| `data/processed/noise_eval/provenance_log.json` | JSON | Audit log | Transformation metadata & random seeds |

---

## 4. Open Risks & Phase 2 Gate Status

1. **Unresolved Risks:**
   - **None detected.** Clean baseline integrity verified via SHA-256; noise rate bounds and evidence retention verified by automated unit tests.
2. **Phase 2 Approval Gate:**
   - In accordance with AGENTS.md rule 4 and scope restrictions, execution is **STOPPED at the Phase 2 gate**.
   - Ready to proceed to **Phase 3: Multi-Strategy Retrieval Pipeline (BM25, Dense Embedding, Cross-Encoder Reranker, Reciprocal Rank Fusion)** upon explicit approval.
