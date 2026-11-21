# Phase 0 Walkthrough: Verification & Environment Setup

## 1. What Was Executed in Phase 0

1. **Environment & Hardware Inspection:**
   - Python 3.10.6 on Windows.
   - Identified CPU-only setup (no CUDA) and available RAM (~2.4 GB).
   - Identified critical C: drive storage constraint (0.84 GB free).
   - Created dedicated project environment in `D:\Indic-QA-RAG\venv` and isolated transliteration environment in `D:\Indic-QA-RAG\venv_xlit`.
   - Set all pip and Hugging Face caches to `D:\` (`HF_HOME=D:\hf_cache`, `TEMP=D:\pip_tmp`, `PIP_CACHE_DIR=D:\pip_cache`).

2. **Repository Architecture Setup:**
   - Created directory skeleton (`docs/walkthroughs/`, `configs/prompts/`, `src/indicrag/`, `tests/`, `results/`, `requirements/`).
   - Configured `pyproject.toml` with Ruff and Pytest settings.
   - Initialized `AGENTS.md`, `.gitignore`, and `.env.example`.
   - Pinned `requirements/base.txt` and `requirements/xlit.txt`.

3. **Dataset & Protocol Verification:**
   - Verified `l3cube-pune/indic-squad` structure on Hugging Face Hub (CC-BY-4.0).
   - Discovered per-language subdirectories (`Hindi/train.json`, `Hindi/val.json`, `Hindi/test.json`).
   - Verified exact schema and sample records via HTTP range requests without downloading full 271 MB file.
   - Confirmed unanswerable questions have `answers: {'text': [], 'answer_start': []}`.
   - Verified 1-to-1 question ID match with SQuAD 2.0 (`rajpurkar/squad_v2`, `is_impossible: True`).
   - Verified RGB protocol and licensing (`CC-BY-NC-SA 4.0` from `chen700564/RGB`). Confirmed clean-room protocol reimplementation policy.

4. **Smoke & Hinglish Suite Execution:**
   - Ran `pytest tests/ -v`:
     - `test_validator_clean_roman`: **PASSED**
     - `test_validator_devanagari_leakage`: **PASSED**
     - `test_validator_missing_digits`: **PASSED**
     - `test_validator_duplicate_detection`: **PASSED**
     - `test_disk_cache_roundtrip`: **PASSED**
     - `test_generator_with_stub_client`: **PASSED**
     - `test_pydantic_state_schema`: **PASSED**
     - `test_langgraph_stub_graph`: **PASSED**
     - `test_budget_enforcement`: **PASSED**
     - `test_no_leakage_stub`: **PASSED**
     - `test_reason_codes`: **PASSED**
     - **Result:** **11 passed in 0.37s**.

---

## 2. Metrics & Environment Facts

- **Tests Passed:** 11 / 11 (100%)
- **Pytest Execution Duration:** 0.37 seconds
- **Memory Consumption during Test:** < 100 MB
- **Disk Space on D Drive:** 264.9 GB available
- **Disk Space on C Drive:** 0.84 GB available (preserved through cache redirection)
- **Hinglish Prompt Version:** `v1.0` (`configs/prompts/hinglish_generation_v1.yaml`)
- **Disk Cache Location:** `.cache/hinglish_gen/`

---

## 3. Open Issues & Gate Status (Phase 0 Complete)

1. **Check 5 Resolved under Option 2:**
   - Few-shot API Hinglish generation pipeline implemented with strict validator, provenance tracking, and persistent disk cache.
   - All 8 user conditions integrated and verified with unit tests.
   - Preserves original Hindi questions completely unchanged.
   - Synthetically generated Hinglish documented as synthetic; planned human annotation retained.
2. **Phase 0 Approval Gate Reached:**
   - Phase 0 verification complete and all tests passing.
   - Stopping execution per Rule 4 and User Condition 8.
   - Ready for Phase 1 (`src/indicrag/data/split.py` and `tests/test_leakage.py`) upon explicit user approval.
