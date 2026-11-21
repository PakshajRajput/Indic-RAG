# Phase 9 — IndicRAG-QA Interactive Research Dashboard Walkthrough

**Phase:** Phase 9 (Interactive Research Dashboard & Operations Platform)  
**Status:** Completed & Fully Verified  
**Date:** October 19, 2026  
**Test Suite:** 84 passed / 84 total (10 new Phase 9 dashboard tests in `tests/test_phase9_dashboard.py`)  
**Design Reference:** [PolarOps](https://polarops-two.vercel.app/) (Antarctic Operational Digital Twin)  

---

## 1. Executive Summary

Phase 9 delivers a production-grade, interactive web application and research operations platform for the **IndicRAG-QA (PJ-09)** project. Drawing aesthetic, informational, and operational inspiration from **PolarOps**, the platform presents IndicRAG-QA as a research-grade, multi-agent AI operations workspace designed for:
1. **Interactive Multi-Script QA:** Real-time query execution supporting Roman-script Hinglish, Devanagari Hindi, and English.
2. **Dynamic Script Awareness:** Generating Roman Hinglish responses for Hinglish queries, Devanagari for Hindi queries, and English for English queries without converting answers into unintended scripts.
3. **Multi-Agent StateGraph Telemetry:** Visualizing node transitions, execution order, retrieval rounds, retry boundaries, and reason codes across Systems S0 through S5.
4. **Evidence & Citation Grounding:** Providing source passage attribution with inline clickable citations `[psg_xxx]` linked to retrieved passages and distractor annotations.
5. **Interactive Evaluation Lab:** Presenting frozen Phase 6 and Phase 8 experimental results, exact counts/denominators, noise degradation curves, cross-script Hinglish ablations, and paired statistical significance tests.

---

## 2. Architecture & Tech Stack

```
d:\Indic-QA-RAG\
├── src\indicrag\web\
│   ├── __init__.py           # Package exports (app, create_app)
│   ├── app.py                # FastAPI REST API & Single-Page Application server
│   └── static\               # PolarOps-inspired frontend assets
│       ├── index.html        # Semantic HTML5 SPA layout with custom components
│       ├── css\
│       │   └── style.css     # Dark polar space theme, glassmorphism, responsive grid
│       └── js\
│           ├── api.js        # Typed API service client
│           ├── charts.js     # Resolution-independent SVG chart renderers
│           └── app.js        # SPA navigation router & state manager
├── scripts\
│   └── run_dashboard.py      # Uvicorn server launcher
└── tests\
    └── test_phase9_dashboard.py # 10 automated unit and integration tests
```

### Backend Engineering (`FastAPI` & `Uvicorn`)
- **FastAPI Core:** Lightweight, asynchronous REST API serving both API endpoints and the static SPA.
- **CorpusManager:** Loads the 2,078-passage Clean Dev corpus and 2,493-passage Perturbed Dev corpus once at startup with in-memory BM25 indexing for low query latency (~3ms).
- **LangGraph Integration:** Connects the interactive query workspace directly to `build_indicrag_graph`, allowing live execution across Systems S0 through S5 with typed Pydantic state (`IndicRAGState`).
- **Strict Judge Isolation:** Enforces that the offline `JudgeAgent` (`google/gemini-2.0-flash`) is never injected into or called by the runtime serving pipeline.

### Frontend Engineering (PolarOps-Inspired Design System)
- **Visual Aesthetic:** Near-black polar navy surfaces (`#070a12`, `#0c1220`, `#111a2e`), subtle cyan/blue borders (`rgba(56, 189, 248, 0.12)`), and high-contrast status pills (`chip-cyan`, `chip-emerald`, `chip-amber`, `chip-rose`).
- **Typography:** Google Fonts pairing `IBM Plex Mono` (crisp monospace metrics and telemetry), `Rajdhani` (technical headers and systems identifiers), and `Inter` (readable body prose).
- **Interactive Componentry:** Collapsible navigation sidebar, breadcrumb topbar, real-time client-side script detector, query runner, citation chips, graph flow diagram, and vector charting.

---

## 3. Application Navigation & Views

### A. Research Operations Overview (`#view-overview`)
- **Executive Metric Cards:**
  - Clean Answerable Accuracy: **$92.00\%$** (Systems S2 / S5 on Dev).
  - Roman Hinglish Baseline: **$2.67\%$** (Method A with $97.33\%$ false refusals).
  - Unanswerable Rejection Rate: **$100.00\%$** ($0.00\%$ Hallucination Rate).
  - Noise Degradation Drop: **$-8.00$ pp** under $20\%$ counterfactual distractors.
- **Interactive Visualizations:**
  - Systems Ladder S0–S5 comparison chart (Clean vs. Perturbed).
  - Cross-Script Hinglish Normalization Deficit horizontal bar chart.
- **Operational Budget & Guardrails:** Visual chips confirming hard limits (max 8 LLM calls, max 2 retrieval rounds, max 1 retry loop, offline judge decoupled).

### B. Hinglish QA Workspace (`#view-workspace`)
- **Multi-Script Query Input:** Large textarea with automatic real-time script classification (`Hinglish (Roman Script)`, `Hindi (Devanagari)`, `English (Latin Script)`).
- **Curated Query Samples:**
  1. *Roman Hinglish (Answerable):* `"Laxmi Mittal ka business kisme hai aur unki company ka naam kya hai?"`
  2. *Devanagari Hindi (Answerable):* `"लक्ष्मी निवास मित्तल का जन्म किस राज्य में हुआ था?"`
  3. *Roman Hinglish (Unanswerable):* `"Laxmi Mittal ne 2025 mein Mars par kaunsi factory banayi thi?"`
  4. *Devanagari Hindi (Unanswerable):* `"लक्ष्मी मित्तल ने मंगल ग्रह पर कौन सी इस्पात मिल स्थापित की थी?"`
  5. *English (Cross-Lingual):* `"What is the primary industry and birthplace of Lakshmi Mittal?"`
- **Execution Controls:** Systems ladder selector (S0 to S5) and Search Corpus selector (Clean Dev vs Perturbed Dev).
- **Grounded Response Viewer:**
  - Displays synthesized final answer with clickable citation chips (e.g., `[psg_99b8ad96e4f6]`).
  - Highlights abstentions with reason code banners (e.g. `ReasonCode: NO_EVIDENCE` or `ANSWERER_REFUSED`).
  - Script-aware response rendering (Roman Hinglish receives Roman Hinglish answers; Hindi receives Hindi; English receives English).
- **Evidence Drawer:** Shows retrieved passages with relevance scores, passage IDs, and counterfactual distractor tags.

### C. Agent Operations & Graph Telemetry (`#view-agents`)
- **Architectural Notice:** Prominent banner confirming offline Judge Agent decoupling.
- **Interactive StateGraph Topology:**
  `Supervisor` $\rightarrow$ `Script Detector` $\rightarrow$ `Query Agent` $\rightarrow$ `Retriever Agent` $\rightarrow$ `Evidence Grader` $\rightarrow$ `Answerer Agent` $\rightarrow$ `Verifier Agent` $\rightarrow$ `Abstention Arbiter`
- **Live Node States:** Dynamic border/glow highlighting (`active`, `success`, `retrying`, `abstention`) reflecting the last executed query.
- **Trace Timeline:** Step-by-step chronological audit log detailing node name, action, retrieval round, retry count, LLM calls, and typed reason codes.

### D. Evidence & Corpus Explorer (`#view-evidence`)
- **Searchable Passage Index:** Real-time keyword and passage ID filtering across 2,078 Clean passages or 2,493 Perturbed passages.
- **Passage Metadata Table:** Shows Passage ID, Article Title, Source Partition, Character Length, and Text Excerpts.
- **Distractor Identification:** Visually tags the 415 counterfactual distractor passages with amber badges.

### E. Evaluation Lab & RGB Benchmark (`#view-eval`)
- **Systems Ladder Matrix:** Tabular comparison of S0 through S5 on Clean and Perturbed conditions across both Dev ($N=150$) and held-out Test ($N=75$) splits.
- **Mathematical Integrity Badge:** Verifies that $\text{Acc}_{\text{ans}} + \text{FRR} + \text{Err}_{\text{ans}} \equiv 1.0$ and $\text{Acc}_{\text{unans}} + \text{HR} \equiv 1.0$ strictly hold across all 32 conditions.
- **Retry Transition Matrix ($N=75$):** Exhaustive accounting showing that on Perturbed Dev, S5 avoided 5 false refusals ($16.00\% \rightarrow 9.33\%$), but all 5 transitioned to incorrect answers ($w_{\text{ans}}=5$), preserving $\text{Acc}_{\text{ans}} = 84.00\%$ ($63/75$).
- **Paired Statistical Significance:** Exact McNemar tests, discordant pairs ($b, c$), raw $p$-values, and Bonferroni adjustments ($\alpha_{\text{adj}} = 0.0167$).

---

## 4. API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Returns system health, seed, hardware environment, and decoupled model specifications. |
| `GET` | `/api/samples` | Returns curated query samples covering Hinglish, Hindi, and English scenarios. |
| `GET` | `/api/metrics/overview` | Returns executive summary statistics from Phase 8 consolidated artifacts. |
| `GET` | `/api/metrics/ladder?split={dev\|test}` | Returns S0–S5 systems ladder metrics for the requested benchmark partition. |
| `GET` | `/api/metrics/noise` | Returns Clean vs. Perturbed noise degradation drops ($\Delta_{\text{pp}}$). |
| `GET` | `/api/metrics/hinglish` | Returns Cross-Script Hinglish normalization ablation (Methods A, B, C). |
| `GET` | `/api/metrics/retry` | Returns S2 $\rightarrow$ S5 paired retry transition matrix on 75 answerable queries. |
| `GET` | `/api/metrics/significance` | Returns McNemar exact tests and bootstrap confidence intervals. |
| `GET` | `/api/evidence/passages` | Searches corpus passages with pagination, keyword search, and corpus toggling. |
| `POST` | `/api/query/execute` | Executes a live query through LangGraph, returning grounded answer, citations, and traces. |
| `GET` | `/{full_path:path}` | Serves the Single-Page Application (SPA) frontend. |

---

## 5. Local Development & Launch Instructions

### Starting the Interactive Dashboard
From the project root directory (`d:\Indic-QA-RAG`):
```powershell
d:\Indic-QA-RAG\venv\Scripts\python scripts/run_dashboard.py --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in any modern web browser.

### Running Phase 9 Automated Tests
```powershell
d:\Indic-QA-RAG\venv\Scripts\pytest tests/test_phase9_dashboard.py -v
```
- **Execution Time:** ~5.2 seconds
- **Result:** 10 passed, 0 failed

### Running the Full Regression Suite (Phases 0–9)
```powershell
d:\Indic-QA-RAG\venv\Scripts\pytest -q
```
- **Execution Time:** ~7.5 seconds
- **Result:** 84 passed, 0 failed (100% green across all 11 test modules)

---

## 6. Verification & Governance Audit

1. **Protocol Manifest Preservation:** `results/phase6_protocol_manifest.json` and all 6 core benchmark splits and search corpora remain byte-identical to their frozen Phase 6 hashes.
2. **Offline Judge Decoupling:** Verified via `test_judge_agent_isolation_in_dashboard` that the offline Judge Agent is never imported, instantiated, or executed within the serving pipeline.
3. **Data Integrity:** All historical metrics in the dashboard come strictly from verified Phase 8 machine-readable outputs in `results/phase8/`. No fake telemetry or synthetic results were seeded.
4. **Refusal vs. Recovery Honesty:** The retry transition matrix strictly preserves the finding that avoided false refusals converted to incorrect answers rather than being misreported as recovered correct answers.
