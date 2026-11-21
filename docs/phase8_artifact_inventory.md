# Phase 8 — Artifact Inventory & Provenance Manifest

This inventory provides an authoritative, auditable catalog of all datasets, corpora, evaluation outputs, derived statistical tables, and publication figures in the IndicRAG-QA repository. Every item is verified against its source provenance and cryptographic hash.

---

## 1. Primary Benchmark Datasets & Search Corpora

| Artifact Path | Partition | Condition | Total Items / Passages | Answerable / Unanswerable Breakdown | SHA-256 Hash | Integrity Status |
|---|---|---|---|---|---|---|
| `data/processed/benchmark_dev_150.json` | Dev | Query Benchmark | 150 queries | 75 answerable / 75 unanswerable | `2e7384498a71936770098a6147577aadefe3391c3ed244942fbf0f202d4922fb` | FROZEN / VERIFIED |
| `data/processed/benchmark_test_75.json` | Test | Query Benchmark | 75 queries | 38 answerable / 37 unanswerable | `532f52986487d9e1f00b9c5872d663eae1ba7d40d1b13fc1cbdf15b538308f6c` | FROZEN / VERIFIED |
| `data/processed/corpus_dev.json` | Dev | Clean Search Corpus | 2,078 passages | N/A (1,039 articles) | `d294086976c71200e004b2976048459b7e22b21f5c89c2e5d937eaebf1596d38` | FROZEN / VERIFIED |
| `data/processed/corpus_test.json` | Test | Clean Search Corpus | 1,061 passages | N/A (529 articles) | `d2749807efec7b76a10bcc0d36958e2ac71847b4b8976504dfb2be37b289687f` | FROZEN / VERIFIED |
| `data/processed/counterfactual/corpus_dev_perturbed.json` | Dev | Perturbed (Noise) | 2,493 passages | 2,078 clean + 415 counterfactual distractors | `ab0ee0f18c879897f20de77142e20662b928484777aedf98d35610d7c567491e` | FROZEN / VERIFIED |
| `data/processed/counterfactual/corpus_test_perturbed.json` | Test | Perturbed (Noise) | 1,481 passages | 1,061 clean + 420 counterfactual distractors | `f069777c41c55059febb24737fabc06245a75e60ef034ac6717ad0689d01dd65` | FROZEN / VERIFIED |

---

## 2. Frozen Protocol & Configuration Artifacts

| Artifact Path | Protocol Version | Seed | Model Family / Embeddings | Key Thresholds | Notes / Scope |
|---|---|---|---|---|---|
| `results/phase6_protocol_manifest.json` | `1.0.0-phase6-freeze` | 42 | Answerer: `meta-llama/Llama-3.1-8B-Instruct`<br>Judge: `google/gemini-2.0-flash`<br>Dense: `paraphrase-multilingual-MiniLM-L12-v2` | S1 Gate: $0.005$<br>Grader: $0.20$<br>Verifier: $0.75$<br>LLM Calls: $\le 8$<br>Rounds: $\le 2$ | Strictly frozen in Phase 6; records all SHA-256 baseline hashes |
| `configs/retrieval_config.yaml` | Production | 42 | BM25 + Dense Multilingual MiniLM | BM25 ($k_1=1.5, b=0.75$), RRF ($c=60, k=20$) | Config-driven retrieval pipeline |
| `configs/agent_config.yaml` | Production | 42 | Multi-agent LangGraph system ladder | Hard budgets: max 8 LLM calls, max 2 rounds | S0 through S5 system specs |

---

## 3. Experimental Output Summaries (Phase 6 & 7)

| Artifact Path | Split / Condition | Format | Description / Primary Content |
|---|---|---|---|
| `results/phase6_dev_ladder_summary.csv` | Dev (Clean & Perturbed) | CSV | Systems ladder (S0–S5) metrics, 95% CIs, F1, citation scores ($N=150$) |
| `results/phase6_dev_ladder_summary.json` | Dev (Clean & Perturbed) | JSON | Machine-readable systems ladder metrics with CI bounds |
| `results/phase6_test_summary.csv` | Test (Clean & Perturbed) | CSV | Held-out Test evaluation across S0–S5 ($N=75$, 38 ans / 37 unans) |
| `results/phase6_test_summary.json` | Test (Clean & Perturbed) | JSON | Machine-readable Test evaluation metrics |
| `results/phase6_hinglish_ablation_summary.csv` | Dev Perturbed | CSV | Roman Hinglish cross-script ablation (Method A vs B vs C) |
| `results/phase6_hinglish_ablation_summary.json` | Dev Perturbed | JSON | Machine-readable Hinglish ablation records |
| `results/phase6_loao_ablation_summary.csv` | Dev Perturbed | CSV | Leave-One-Agent-Out ablation records for S5 components |
| `results/phase6_loao_ablation_summary.json` | Dev Perturbed | JSON | Machine-readable LOAO ablation records |
| `results/phase6_significance_report.json` | Dev | JSON | McNemar exact tests and paired bootstrap differences |
| `results/phase6_cost_summary.json` | Dev + Test | JSON | Token consumption, API cost accounting, and call counts |
| `results/phase7_traces_sample.jsonl` | Dev | JSONL | Full execution traces across all 8 LangGraph nodes |
| `results/phase7_records_sample.csv` | Dev | CSV | Detailed per-query evaluation records with reason codes |
| `results/phase7_aggregate_report.json` | Dev | JSON | Aggregate evaluation report validating RGB metric identities |

---

## 4. Phase 8 Derived Statistical Tables & Machine-Readable Deliverables

All derived artifacts are located in `results/phase8/` and reproducible via `scripts/run_phase8_analysis.py`:

| Deliverable Path | Description | Denominators / Sample Size | Key Metrics Reported |
|---|---|---|---|
| `results/phase8/table1_systems_ladder_dev.csv` | Dev Systems Ladder | $N_{\text{total}}=150, N_{\text{ans}}=75, N_{\text{unans}}=75$ | $\text{Acc}_{\text{ans}}, \text{FRR}, \text{Err}_{\text{ans}}, \text{Acc}_{\text{unans}}, \text{HR}, \text{Acc}_{\text{overall}}$, F1 |
| `results/phase8/table2_systems_ladder_test.csv` | Test Systems Ladder | $N_{\text{total}}=75, N_{\text{ans}}=38, N_{\text{unans}}=37$ | $\text{Acc}_{\text{ans}}, \text{FRR}, \text{Err}_{\text{ans}}, \text{Acc}_{\text{unans}}, \text{HR}, \text{Acc}_{\text{overall}}$, F1 |
| `results/phase8/table3_clean_vs_perturbed.csv` | Noise Degradation Comparison | Dev ($N=150$) and Test ($N=75$) | $\Delta\text{Acc}_{\text{ans}}$ (pp), $\Delta\text{FRR}$ (pp), F1 clean vs perturbed |
| `results/phase8/table4_hinglish_ablation.csv` | Hinglish Cross-Script Ablation | $N_{\text{total}}=150, N_{\text{ans}}=75, N_{\text{unans}}=75$ | Method A vs B vs C: $\text{Acc}_{\text{ans}}, \text{FRR}, \text{Acc}_{\text{overall}}$, F1 |
| `results/phase8/table5_loao_ablation.csv` | Leave-One-Agent-Out Ablation | Dev Perturbed ($N=150$) | Full S5 vs w/o QueryAgent, Grader, Verifier, Arbiter |
| `results/phase8/table6_statistical_significance.csv` | Paired Hypothesis Tests | Paired comparisons ($N=150$) | McNemar $b, c, p_{\text{raw}}$, Bonferroni threshold ($\alpha=0.0167$) |
| `results/phase8/table7_retry_transition_matrix.csv` | S2 $\rightarrow$ S5 Retry Transition Matrix | $N_{\text{ans}}=75$ (Perturbed Dev) | Transition counts: 63 correct, 7 refusals, 5 incorrect |
| `results/phase8/metric_integrity_audit.json` | Numerical Identity Audit | 32 evaluated conditions | Verification that RGB identities sum to $1.0000 \pm 10^{-5}$ |
| `results/phase8/phase8_consolidated_summary.json` | Consolidated Master Deliverable | Complete project | All tables, manifest audit, and significance test summaries |

---

## 5. Phase 8 Publication-Ready Vector Figures

All figures are resolution-independent XML vector graphics (`.svg`) suitable for direct inclusion in LaTeX publications:

| Figure Path | Title | Description |
|---|---|---|
| `results/phase8/fig1_systems_ladder_accuracy.svg` | Systems Ladder Accuracy (Dev Set) | Dual-bar chart comparing Clean vs Perturbed Dev accuracy across S0–S5 |
| `results/phase8/fig2_clean_vs_perturbed_drop.svg` | Noise Degradation Profile | Drop in answerable accuracy (pp) caused by counterfactual distractors |
| `results/phase8/fig3_hinglish_cross_script_gap.svg` | Cross-Script Retrieval Gap | Contrast between Devanagari Hindi baseline and Roman Hinglish methods |
| `results/phase8/fig4_retry_transition_breakdown.svg` | Paired Retry Transition Breakdown | Visual flow of the 75 answerable queries from S2 to S5 on Perturbed Dev |
