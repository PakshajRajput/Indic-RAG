# Phase 1 Walkthrough: Data Ingestion and Leakage-Safe Splitting

## 1. What Was Executed in Phase 1

1. **Data Ingestion & Integrity Audit:**
   - Ingested IndicSQuAD Hindi data (`Hindi_val.json`, 11,873 rows; `Hindi_test.json`, 11,803 rows) from `data/raw/`.
   - Verified exact schema: `['id', 'title', 'context', 'question', 'answers']`.
   - Verified answerability representation: 5,928 answerable (49.9%) and 5,945 unanswerable (50.1%) in `Hindi_val.json`.
   - Verified negative question format: `answers: {'text': [], 'answer_start': []}` with `is_impossible: True`.

2. **Connected-Component Cluster Analysis (Crucial Research Finding):**
   - **Research finding:** Title deduplication alone is insufficient. Empirical inspection revealed cross-article content sharing:
     - `विदेश` and `विदेश महाविद्यालय` shared context paragraphs.
     - `कुँवारी` and `आर्थिक असमानता` shared context paragraphs.
     - `फ्रेस्नो, कैलिफ़ोर्निया` and `दक्षिणी कैलिफोर्निया` shared identical question text (`लॉस एंजिल्स की जनसंख्या कितनी है?`).
   - Built a graph-based connected-component clustering algorithm in `src/indicrag/data/split.py` that links articles sharing contexts or questions into atomic, unbreakable clusters.
   - Identified **35 independent clusters** across the 38 titles (32 single-article clusters, 3 multi-article linked clusters).

3. **Deterministic Partitioning with Fixed Seed (`seed=42`):**
   - Partitioned the 35 clusters deterministically into:
     - **Train:** 27 clusters (28 titles, 9,010 records)
     - **Dev:** 4 clusters (5 titles, 1,361 records)
     - **Test:** 4 clusters (5 titles, 1,502 records)
   - Extracted evaluation benchmark subsets (Proposal A):
     - `benchmark_dev_150.json`: Exactly 150 questions (75 answerable + 75 unanswerable).
     - `benchmark_test_75.json`: Exactly 75 questions (38 answerable + 37 unanswerable).
   - Preserved original IDs and linked synthetic Hinglish via `synthetic_hinglish_id: f"{id}_hinglish"`.

4. **Comprehensive Leakage Testing (`tests/test_leakage.py`):**
5. **Retrieval Corpus Protocol vs. Question-Partition Protocol:**
   - **Question-Partition Protocol:**
     - Query sets are partitioned by connected clusters into Train, Dev, and Test.
     - Dev evaluation queries (150 items) come exclusively from Dev clusters.
     - Test evaluation queries (75 items) come exclusively from Test clusters.
   - **Search Corpus Protocol:**
     - How answerable queries retrieve gold evidence:
       - Search Corpus for Dev: $\text{Corpus}(\text{Dev}) = \text{Train Passages} \cup \text{Dev Passages}$ (saved to `data/processed/corpus_dev.json`).
       - Search Corpus for Test: $\text{Corpus}(\text{Test}) = \text{Train Passages} \cup \text{Test Passages}$ (saved to `data/processed/corpus_test.json`).
       - Background Corpus: $\text{Corpus}(\text{Train})$ (saved to `data/processed/corpus_background.json`).
     - **Gold Evidence Verification:** Every answerable question in Dev/Test is guaranteed to have its gold passage and answer substring present in its corresponding searchable corpus.
     - **Corpus Isolation:** `corpus_dev.json` contains zero passages from Test articles; `corpus_test.json` contains zero passages from Dev articles.
     - **Zero Annotation Leakage:** All corpus passages are stripped of labels (`question`, `answers`, `is_impossible`, `synthetic_hinglish_id`). They are unannotated text chunks with only `passage_id`, `title`, `text`, and `source_partition`.

6. **Comprehensive Leakage & Corpus Testing (`tests/test_leakage.py`):**
   - Implemented 11 automated leakage, balance, and corpus validation tests:
     - `test_zero_title_leakage`: **PASSED** (0 overlapping titles)
     - `test_zero_normalized_title_leakage`: **PASSED** (0 normalized title overlaps)
     - `test_zero_context_leakage`: **PASSED** (0 overlapping context paragraphs)
     - `test_zero_question_leakage`: **PASSED** (0 overlapping question strings)
     - `test_zero_id_leakage`: **PASSED** (0 overlapping IDs)
     - `test_benchmark_subset_balance`: **PASSED** (exact 75/75 dev and 38/37 test balance)
     - `test_record_schema_and_hinglish_link`: **PASSED** (provenance keys preserved)
     - `test_connected_component_clustering`: **PASSED** (unit test on cluster grouping)
     - `test_gold_evidence_present_in_search_corpus`: **PASSED** (100% of answerable Dev and Test queries have gold context & answer in corpus)
     - `test_corpus_isolation_and_no_annotation_leakage`: **PASSED** (zero cross-partition title leakage in corpus; zero QA labels in corpus)
     - `test_qualified_leakage_assertion_scope`: **PASSED** (explicit verification of checked fields)

---

## 2. Test Suite Execution & Metrics

Executed via `D:\Indic-QA-RAG\venv\Scripts\python.exe -m pytest tests/ -v`:

| Test Module | Tests | Status | Execution Time |
|---|---|---|---|
| `tests/test_leakage.py` | 11 | **PASSED** ✅ | 1.19s |
| `tests/test_hinglish_pipeline.py` | 6 | **PASSED** ✅ | 0.12s |
| `tests/test_phase0_smoke.py` | 5 | **PASSED** ✅ | 0.45s |
| **Total Test Suite** | **22 / 22** | **PASSED** (100%) | **1.76s** |

### Partition Audit Metrics (`data/processed/split_metadata.json`)

```json
{
  "seed": 42,
  "total_raw_records": 11873,
  "total_clusters": 35,
  "train_clusters": 27,
  "dev_clusters": 4,
  "test_clusters": 4,
  "train_records": 9010,
  "dev_records": 1361,
  "test_records": 1502,
  "benchmark_dev_sampled": 150,
  "benchmark_test_sampled": 75,
  "zero_title_leakage": true,
  "zero_context_leakage": true,
  "zero_question_leakage": true,
  "zero_id_leakage": true
}
```

---

## 3. Allocation of Articles Across Partitions

| Partition | Clusters | Articles (Titles) | Total Records | Benchmark Sample | Search Corpus Passages |
|---|---|---|---|---|---|
| **Dev** | 4 | पैकेट चंडी, भूगर्भशास्त्र, साम्राज्यवाद, [विदेश + विदेश महाविद्यालय] | 1,361 | 150 (75 ans, 75 unans) | 2,139 (Train + Dev) |
| **Test** | 4 | कम्प्यूटेशनल_जटिलता_सिद्धांत, जैक्सनविले_फ्लोरिडा, बल, [आर्थिक असमानता + कुँवारी] | 1,502 | 75 (38 ans, 37 unans) | 2,176 (Train + Test) |
| **Train** | 27 | राइन, वारसा, भाप का इंजन, युआन वंश, ऑक्सीजन, [दक्षिणी कैलिफोर्निया + फ्रेस्नो], ... | 9,010 | Full background pool | 1,818 (Train only) |

---

## 4. Qualification of Leakage Assertions & Gate Status

1. **Qualification of Leakage Scope:**
   - **Verified Ground Truth:** Zero exact and normalized overlap has been mathematically proven and tested for:
     (1) Article title strings, (2) Normalized alphanumeric title strings, (3) Context paragraph texts, (4) Question strings, and (5) Sample IDs.
   - **Research Qualification:** These automated tests confirm exact lexical and content isolation for the checked fields. They do **not** imply universal semantic independence across general world-knowledge encyclopedic concepts (e.g. universal historical or scientific principles that exist across Wikipedia articles).
2. **Phase 1 Approval Gate:**
   - In accordance with AGENTS.md rule 4 and user scope restrictions, execution is **STOPPED at the Phase 1 gate**.
   - Ready to proceed to **Phase 2** (Corpus Preparation, Noise Injection per RGB protocol, and Counterfactual Distractor generation) upon your explicit approval.
