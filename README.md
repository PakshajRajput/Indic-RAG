# IndicRAG-QA: Multi-Agent, Abstention-Aware RAG for Hindi & Hinglish

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3106/)
[![Test Suite](https://img.shields.io/badge/pytest-84%20passed-brightgreen.svg)](tests/)
[![Benchmark](https://img.shields.io/badge/protocol-RGB%20AAAI%202024-orange.svg)](docs/final_project_handoff.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An implementation and rigorous benchmark of a **multi-agent, abstention-aware Retrieval-Augmented Generation (RAG)** system for low-resource and code-mixed South Asian QA: **Hindi (Devanagari script)** and **Hinglish (Roman script)**. The project follows the evaluation methodology of the RGB benchmark ([Chen et al., AAAI 2024](https://arxiv.org/abs/2309.01431)).

---

## Key Features

1. **Multi-Agent Orchestration (LangGraph):**
   - **Script & Language Detector:** Unicode block distribution classifier (Devanagari vs. Roman Hinglish).
   - **Query Reformulation Agent:** Roman-to-Devanagari transliteration and retry query rewriting.
   - **Multi-Strategy Hybrid Retriever:** Reciprocal Rank Fusion (RRF $c=60$) over multilingual BM25 and multilingual dense embeddings (`paraphrase-multilingual-MiniLM-L12-v2`), reranked by `mmarco-mMiniLMv2-L12-Hundo`.
   - **Evidence Grader:** Topical relevance and counterfactual noise filtering.
   - **Grounded Answer Generator:** Strict synthesis with citation tags (`[psg_xxx]`).
   - **Claim Verifier:** Token- and claim-level grounding and citation fidelity audit.
   - **Abstention Arbiter:** Calibrated abstention with bounded retry dynamics.
   - **Supervisor:** Hard budget enforcement (max tokens, latency, loop caps).

2. **RGB Evaluation Protocol & Zero-Fabrication Governance:**
   - Evaluated along four core dimensions: **Noise Robustness**, **Negative Rejection (Abstention)**, **Information Integration**, and **Counterfactual Robustness**.
   - Strict leak-safe splitting: Train/Dev/Test partitions clustered by article entities to eliminate cross-split context and lexical leakage.
   - Offline evaluation using an independent Judge Agent family (`google/gemini-2.0-flash`) strictly separated from the serving pipeline (`meta-llama/Llama-3.1-8B-Instruct`).
   - Every metric is verifiable from persistent machine-readable artifacts in `results/`.

3. **Interactive Web Dashboard:**
   - FastAPI backend and responsive single-page application (SPA).
   - Live query testing (Hindi Devanagari & Roman Hinglish) with real-time multi-agent state graphs, citations, and evidence inspectability.
   - Systems ladder comparisons, noise degradation curves, and evidence search explorer.

---

## Systems Ladder Results (Dev Benchmark, $N=150$)

Source: [`results/phase8/table1_systems_ladder_dev.csv`](results/phase8/table1_systems_ladder_dev.csv)

| System | Description | Clean Ans Acc ($\%$) | Perturbed Ans Acc ($\%$) | Unans Acc ($\%$) | Hallucination Rate ($\%$) | Overall Acc ($\%$) |
|---|---|---|---|---|---|---|
| **S0** | Naive Single-Pass RAG | 90.67% | 84.00% | 0.00% | 100.00% | 45.33% |
| **S1** | Score-Gated RAG | 90.67% | 84.00% | 4.00% | 96.00% | 47.33% |
| **S2** | Grader + Verifier | 86.67% | 84.00% | 100.00% | 0.00% | 93.33% |
| **S3** | + Query Reformulation | 86.67% | 84.00% | 100.00% | 0.00% | 93.33% |
| **S4** | + Script-Aware Routing | 86.67% | 84.00% | 100.00% | 0.00% | 93.33% |
| **S5** | **Full Multi-Agent + Retry** | **86.67%** | **84.00%** | **100.00%** | **0.00%** | **93.33%** |

### Held-Out Test Evaluation ($N=75$)
Source: [`results/phase8/table2_systems_ladder_test.csv`](results/phase8/table2_systems_ladder_test.csv)
- **Clean Test Accuracy (Answerable):** 86.84% (33/38)
- **Perturbed Test Accuracy (Answerable):** 81.58% (31/38)
- **Unanswerable Accuracy:** 100.00% (37/37) — zero hallucinations on unanswerable inputs.
- **Overall Clean / Perturbed Accuracy:** 93.33% (70/75) / 90.67% (68/75)

### Key Empirical Findings
- **Counterfactual Noise Drop:** $\Delta_{\text{pp}} = -2.67\text{ pp}$ on Dev ([`results/phase8/table3_clean_vs_perturbed.csv`](results/phase8/table3_clean_vs_perturbed.csv)), absorbed entirely via safe abstention rather than hallucinated misinformation.
- **Hinglish Cross-Script Retrieval Gap:** Roman-script queries against native Devanagari corpus collapse to $2.67\%$ unnormalized accuracy and $0.00\%$ under rule-based transliteration ([`results/phase8/table4_hinglish_ablation.csv`](results/phase8/table4_hinglish_ablation.csv)), demonstrating that heuristic phonetic mapping is insufficient for colloquial code-mixed queries.
- **Leave-One-Agent-Out (LOAO) Ablation:** Removing either QueryAgent reformulation or Arbiter retry increases the False Refusal Rate (FRR) from $9.33\%$ to $16.00\%$ on perturbed inputs ([`results/phase8/table5_loao_ablation.csv`](results/phase8/table5_loao_ablation.csv)).

---

## Repository Structure

```
.
├── configs/                     # YAML configuration files (models, retrieval, thresholds)
│   ├── noise_config.yaml
│   ├── retrieval_config.yaml
│   └── prompts/                 # Agent system prompts
├── data/
│   ├── sample_answerable_dev.json
│   └── (raw/ and processed/ gitignored as per reproducibility protocol)
├── docs/                        # Complete project documentation & phase walkthroughs
│   ├── final_project_handoff.md
│   ├── phase8_statistical_analysis.md
│   ├── VERIFICATION_REPORT.md
│   └── walkthroughs/            # Detailed Phase 0 to Phase 9 reports
├── notebooks/                   # Exploration & visualization notebooks
├── paper/                       # Research paper source & LaTeX templates
├── requirements/                # Pinned dependency manifests
│   ├── base.txt
│   └── xlit.txt
├── results/                     # Machine-readable evaluation outputs & figures
│   ├── phase8/                  # Publication-ready tables (CSV/JSON) & vector SVG charts
│   ├── retrieval_benchmark.csv
│   └── phase6_dev_ladder_summary.csv
├── scripts/                     # Executable scripts (analysis, evaluation, dashboard)
│   ├── run_dashboard.py
│   ├── run_phase8_analysis.py
│   └── generate_phase7_artifacts.py
├── src/indicrag/                # Core Python package
│   ├── agents/                  # LangGraph multi-agent architecture (S0–S5)
│   ├── data/                    # Leak-safe splitting and noise injection
│   ├── eval/                    # RGB metrics, judge agent, statistical tests
│   ├── generation/              # Cached LLM client & audit logger
│   ├── hinglish/                # Transliteration and script validation
│   ├── retrieval/               # BM25, dense search, reranker, RRF
│   └── web/                     # FastAPI server and modern web dashboard
└── tests/                       # 84 automated pytest test suites
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10.x
- Git

### 2. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/PakshajRajput/Indic-RAG.git
cd Indic-RAG

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements/base.txt
pip install -e .
```

### 3. Environment Variables
Copy `.env.example` to `.env` and fill in your keys (optional for running stubs/unit tests; required for hosted generation):
```powershell
cp .env.example .env
```

---

## Running Tests & Reproducing Results

### 1. Run Complete Test Suite (84 Passing Tests)
```powershell
.\venv\Scripts\pytest.exe
```

### 2. Regenerate Phase 8 Statistical Analysis & Charts
```powershell
.\venv\Scripts\python.exe scripts/run_phase8_analysis.py
```

### 3. Launch Interactive Web Dashboard
```powershell
.\venv\Scripts\python.exe scripts/run_dashboard.py
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Citation & References

```bibtex
@inproceedings{chen2024benchmarking,
  title={Benchmarking Large Language Models in Retrieval-Augmented Generation},
  author={Chen, Jiawei and Lin, Hongyu and Han, Xianpei and Sun, Le},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={17754--17762},
  year={2024}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
