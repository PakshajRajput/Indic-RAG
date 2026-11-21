# Phase 8 — Statistical Analysis & Experimental Results Validation

**Date:** October 19, 2026  
**Audience:** Paper Author / Technical Evaluation Committee  
**Protocol Version:** `1.0.0-phase6-freeze`  
**Evaluation Standard:** RGB Benchmark Protocol (Chen et al., AAAI 2024)

---

## 1. Evaluation Protocol & Metric Definitions

Following the RGB benchmark framework (Chen et al., AAAI 2024), evaluation queries are strictly partitioned into **Answerable** ($Q_{\text{ans}}$) and **Unanswerable** ($Q_{\text{unans}}$) subsets. All metrics use explicit, non-overlapping numerators and denominators:

- **Answerable Query Set ($N_{\text{ans}}$):**
  - $c_{\text{ans}}$: Correct answer supported by gold evidence and valid citation.
  - $r_{\text{ans}}$: False refusal (system refused or abstained when evidence was present).
  - $w_{\text{ans}}$: Incorrect answer (system gave an ungrounded, incorrect, or contradictory answer).
  - **Answerable Accuracy:** $\text{Acc}_{\text{ans}} = \frac{c_{\text{ans}}}{N_{\text{ans}}}$
  - **False Refusal Rate:** $\text{FRR} = \frac{r_{\text{ans}}}{N_{\text{ans}}}$
  - **Incorrect Answer Rate:** $\text{Err}_{\text{ans}} = \frac{w_{\text{ans}}}{N_{\text{ans}}}$
  - **Fundamental Answerable Identity:**
    $$\text{Acc}_{\text{ans}} + \text{FRR} + \text{Err}_{\text{ans}} \equiv 1.0$$

- **Unanswerable Query Set ($N_{\text{unans}}$):**
  - $c_{\text{unans}}$: Correct rejection/abstention (system recognized that the context does not contain the answer).
  - $h_{\text{unans}}$: Hallucination (system generated an answer despite zero gold support).
  - **Unanswerable Accuracy (Rejection Accuracy):** $\text{Acc}_{\text{unans}} = \frac{c_{\text{unans}}}{N_{\text{unans}}}$
  - **Hallucination Rate:** $\text{HR} = \frac{h_{\text{unans}}}{N_{\text{unans}}}$
  - **Fundamental Unanswerable Identity:**
    $$\text{Acc}_{\text{unans}} + \text{HR} \equiv 1.0$$

- **Overall Benchmark Metrics:**
  - $\text{Acc}_{\text{overall}} = \frac{c_{\text{ans}} + c_{\text{unans}}}{N_{\text{ans}} + N_{\text{unans}}}$
  - $\text{Rejection Rate} = \frac{r_{\text{ans}} + c_{\text{unans}}}{N_{\text{ans}} + N_{\text{unans}}}$
  - **Token F1 & Exact Match (EM):** Macro-averaged token-level overlap against gold answer spans.

---

## 2. S0–S5 Systems Ladder Performance

The systems ladder evaluates incremental architectural components under identical prompt configurations and random seed ($42$):
- **S0:** Single-pass RAG (baseline retriever + generator).
- **S1:** Score-Gated RAG (abstains if top retrieval score $< 0.005$).
- **S2:** Verifier RAG (Answerer + Verifier checking claim overlap $\ge 0.75$).
- **S3:** Graded Verifier RAG (Evidence Grader filtering passages $< 0.20$ + Verifier).
- **S4:** Query-Adapted RAG (Script Detector + Query Agent + Grader + Verifier).
- **S5:** Full Multi-Agent System (Query Agent + Grader + Verifier + Abstention Arbiter with bounded retry).

### 2.1 Development Set ($N=150$, $N_{\text{ans}}=75$, $N_{\text{unans}}=75$)

Source file: [`results/phase8/table1_systems_ladder_dev.csv`](file:///d:/Indic-QA-RAG/results/phase8/table1_systems_ladder_dev.csv)

| System ID | System Name | Corpus Condition | $c_{\text{ans}}$ | $r_{\text{ans}}$ | $w_{\text{ans}}$ | $\text{Acc}_{\text{ans}}$ [95% CI] | $\text{FRR}$ | $\text{Err}_{\text{ans}}$ | $c_{\text{unans}}$ | $h_{\text{unans}}$ | $\text{Acc}_{\text{unans}}$ | $\text{Acc}_{\text{overall}}$ [95% CI] | Token F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **S0** | Single-pass RAG | Clean Dev | 68 | 7 | 0 | 90.67% [84.00%, 97.33%] | 9.33% | 0.00% | 75 | 0 | 100.0% | 95.33% [92.00%, 98.67%] | 0.7377 |
| **S0** | Single-pass RAG | Perturbed Dev | 68 | 7 | 0 | 90.67% [84.00%, 97.33%] | 9.33% | 0.00% | 75 | 0 | 100.0% | 95.33% [92.00%, 98.67%] | 0.7377 |
| **S1** | Score-Gated RAG | Clean Dev | 68 | 7 | 0 | 90.67% [84.00%, 97.33%] | 9.33% | 0.00% | 75 | 0 | 100.0% | 95.33% [92.00%, 98.67%] | 0.7377 |
| **S1** | Score-Gated RAG | Perturbed Dev | 68 | 7 | 0 | 90.67% [84.00%, 97.33%] | 9.33% | 0.00% | 75 | 0 | 100.0% | 95.33% [92.00%, 98.67%] | 0.7377 |
| **S2** | Verifier RAG | Clean Dev | 65 | 10 | 0 | 86.67% [78.67%, 93.33%] | 13.33% | 0.00% | 75 | 0 | 100.0% | 93.33% [88.67%, 96.68%] | 0.6999 |
| **S2** | Verifier RAG | Perturbed Dev | 63 | 12 | 0 | 84.00% [76.00%, 92.00%] | 16.00% | 0.00% | 75 | 0 | 100.0% | 92.00% [87.33%, 96.00%] | 0.6777 |
| **S3** | Graded Verifier RAG | Clean Dev | 65 | 10 | 0 | 86.67% [78.67%, 93.33%] | 13.33% | 0.00% | 75 | 0 | 100.0% | 93.33% [88.67%, 96.68%] | 0.6999 |
| **S3** | Graded Verifier RAG | Perturbed Dev | 63 | 12 | 0 | 84.00% [76.00%, 92.00%] | 16.00% | 0.00% | 75 | 0 | 100.0% | 92.00% [87.33%, 96.00%] | 0.6777 |
| **S4** | Query-Adapted RAG | Clean Dev | 65 | 10 | 0 | 86.67% [78.67%, 93.33%] | 13.33% | 0.00% | 75 | 0 | 100.0% | 93.33% [88.67%, 96.68%] | 0.6999 |
| **S4** | Query-Adapted RAG | Perturbed Dev | 63 | 12 | 0 | 84.00% [76.00%, 92.00%] | 16.00% | 0.00% | 75 | 0 | 100.0% | 92.00% [87.33%, 96.00%] | 0.6777 |
| **S5** | Full Multi-Agent | Clean Dev | 65 | 7 | 3 | 86.67% [78.67%, 93.33%] | 9.33% | 4.00% | 75 | 0 | 100.0% | 93.33% [88.67%, 96.68%] | 0.6999 |
| **S5** | Full Multi-Agent | Perturbed Dev | 63 | 7 | 5 | 84.00% [76.00%, 92.00%] | 9.33% | 6.67% | 75 | 0 | 100.0% | 92.00% [87.33%, 96.00%] | 0.6777 |

### 2.2 Held-Out Test Set ($N=75$, $N_{\text{ans}}=38$, $N_{\text{unans}}=37$)

Source file: [`results/phase8/table2_systems_ladder_test.csv`](file:///d:/Indic-QA-RAG/results/phase8/table2_systems_ladder_test.csv)

| System ID | Corpus Condition | $c_{\text{ans}}$ | $r_{\text{ans}}$ | $w_{\text{ans}}$ | $\text{Acc}_{\text{ans}}$ [95% CI] | $\text{FRR}$ | $\text{Err}_{\text{ans}}$ | $c_{\text{unans}}$ | $h_{\text{unans}}$ | $\text{Acc}_{\text{unans}}$ | $\text{Acc}_{\text{overall}}$ [95% CI] | Token F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **S0–S5** | Clean Test | 33 | 5 | 0 | 86.84% [76.32%, 97.37%] | 13.16% | 0.00% | 37 | 0 | 100.0% | 93.33% [88.00%, 98.67%] | 0.6724 |
| **S0–S5** | Perturbed Test | 31 | 7 | 0 | 81.58% [68.42%, 92.11%] | 18.42% | 0.00% | 37 | 0 | 100.0% | 90.67% [84.00%, 96.00%] | 0.6338 |

---

## 3. Clean vs. Perturbed Corpus Degradation Analysis

Source file: [`results/phase8/table3_clean_vs_perturbed.csv`](file:///d:/Indic-QA-RAG/results/phase8/table3_clean_vs_perturbed.csv)

The injection of 415 counterfactual distractors in Dev and 420 in Test models real-world noisy retrieval conditions.

| Partition | Systems | Clean $\text{Acc}_{\text{ans}}$ | Perturbed $\text{Acc}_{\text{ans}}$ | $\Delta\text{Acc}_{\text{ans}}$ (pp) | Clean $\text{FRR}$ | Perturbed $\text{FRR}$ | $\Delta\text{FRR}$ (pp) | Clean Token F1 | Perturbed Token F1 |
|---|---|---|---|---|---|---|---|---|---|
| **Dev ($N=150$)** | S0, S1 | 90.67% | 90.67% | $0.00\text{ pp}$ | 9.33% | 9.33% | $0.00\text{ pp}$ | 0.7377 | 0.7377 |
| **Dev ($N=150$)** | S2, S3, S4 | 86.67% | 84.00% | **$-2.67\text{ pp}$** | 13.33% | 16.00% | **$+2.67\text{ pp}$** | 0.6999 | 0.6777 |
| **Dev ($N=150$)** | S5 | 86.67% | 84.00% | **$-2.67\text{ pp}$** | 9.33% | 9.33% | $0.00\text{ pp}$ | 0.6999 | 0.6777 |
| **Test ($N=75$)** | S0–S5 | 86.84% | 81.58% | **$-5.26\text{ pp}$** | 13.16% | 18.42% | **$+5.26\text{ pp}$** | 0.6724 | 0.6338 |

### Key Observations:
1. **Unanswerable Robustness:** In all systems across both Dev and Test partitions, unanswerable accuracy remains exactly **$100.0\%$** ($75/75$ on Dev, $37/37$ on Test). Not a single counterfactual distractor induced a hallucinated answer ($\text{HR} = 0.0\%$).
2. **Noise Degradation on Answerable Queries:** Counterfactual distractors degrade answerable accuracy by **$-2.67\text{ pp}$** on Dev ($N_{\text{ans}}=75$, 2 queries displaced) and **$-5.26\text{ pp}$** on Test ($N_{\text{ans}}=38$, 2 queries displaced).
3. **Failure Mechanism:** The degradation manifests as increased false refusals ($\text{FRR}$) rather than hallucinations or false acceptances. When distractors crowd the top-5 retrieval window, gold evidence falls below the top-5 cut, triggering explicit, calibrated abstention.

---

## 4. Hinglish Cross-Script Normalization Ablation

Source file: [`results/phase8/table4_hinglish_ablation.csv`](file:///d:/Indic-QA-RAG/results/phase8/table4_hinglish_ablation.csv)

Evaluated on the synthetic Roman Hinglish benchmark ($N=150$, 75 answerable / 75 unanswerable) against the Hindi search corpus:

| Method | Description | $\text{Acc}_{\text{ans}}$ | $\text{FRR}$ | $\text{Err}_{\text{ans}}$ | $\text{Acc}_{\text{unans}}$ | $\text{Acc}_{\text{overall}}$ [95% CI] | Token F1 |
|---|---|---|---|---|---|---|---|
| **Devanagari Baseline (S2)** | Native Devanagari query | 86.67% | 13.33% | 0.00% | 100.0% | 93.33% [88.67%, 96.68%] | 0.6999 |
| **Method A: Unnormalized** | Raw Roman Hinglish to BM25/Dense | 2.67% (2/75) | 97.33% (73/75) | 0.00% | 100.0% | 51.33% [43.33%, 59.33%] | 0.0196 |
| **Method B: Rule-based Transliteration** | Deterministic reverse-phonetic map | 0.00% (0/75) | 100.0% (75/75) | 0.00% | 100.0% | 50.00% [42.00%, 58.00%] | 0.0000 |
| **Method C: Full Multi-Agent S5** | Script Detector + Query Agent + Arbiter | 0.00% (0/75) | 100.0% (75/75) | 0.00% | 100.0% | 50.00% [42.00%, 58.00%] | 0.0000 |

### Critical Research Finding & Governance Limitation:
- **Heuristic Transliteration Failure:** Rule-based dictionary transliteration (Method B) and zero-shot query adaptation (Method C) fail to bridge the cross-script retrieval gap against Devanagari text ($0.0\%$ answerable accuracy).
- **Phonetic Ambiguity:** Roman Hinglish exhibits severe orthographic variability (e.g., `"kya"`, `"kia"`, `"kiya"`, `"shuru"`, `"shuroo"`), which rigid rule-based tables distort, corrupting BM25 lexical matching.
- **Scientific Conclusion:** Cross-script Indic retrieval cannot be resolved by lightweight heuristic normalization. It remains an **unresolved research limitation** requiring either trained neural sequence-to-sequence transliteration models (e.g., IndicXlit) or dual-indexed multilingual vector representations.

---

## 5. Paired Statistical Significance Testing

Source file: [`results/phase8/table6_statistical_significance.csv`](file:///d:/Indic-QA-RAG/results/phase8/table6_statistical_significance.csv)

Paired tests were executed across identical query instances to evaluate whether observed empirical differences are statistically distinguishable from chance:
- **McNemar's Exact Test:** Evaluates discordant query pairs where System 1 is correct and System 2 is incorrect ($b$), versus System 1 incorrect and System 2 correct ($c$).
- **Paired Bootstrap Difference:** 1,000 resamples computing the 95% confidence interval of the mean accuracy difference.
- **Multiple Comparisons Adjustment:** Bonferroni threshold $\alpha_{\text{Bonferroni}} = \frac{0.05}{3} = 0.0167$.

| Hypothesis / Comparison | Systems Compared | Discordant Pairs ($b, c$) | Total Discordant ($b+c$) | McNemar $p_{\text{raw}}$ | Bonferroni Adjusted $\alpha$ | Statistically Significant? | Mean Gain (95% CI) |
|---|---|---|---|---|---|---|---|
| **$H_1$: Rule Transliteration vs Raw Roman** | Method B vs. Method A | $b=0, c=2$ | 2 | $0.5000$ | $0.0167$ | **No** ($p \ge 0.05$) | $-1.33\text{ pp}$ $[-3.33\%, 0.00\%]$ |
| **$H_2$: Verifier vs. Score Gate** | S2 vs. S1 (Clean Dev) | $b=0, c=3$ | 3 | $0.2500$ | $0.0167$ | **No** ($p \ge 0.05$) | $-2.00\text{ pp}$ (empirical difference) |
| **$H_3$: Full Multi-Agent vs Single-Pass** | S5 vs. S0 (Clean Dev) | $b=0, c=3$ | 3 | $0.2500$ | $0.0167$ | **No** ($p \ge 0.05$) | $-2.00\text{ pp}$ (empirical difference) |

### Governance Note:
None of the pairwise comparisons achieve statistical significance at $\alpha = 0.05$ or $\alpha_{\text{Bonferroni}} = 0.0167$. All reported performance differences must be documented strictly as **empirical observations on this sample**, not as proven superiorities.

---

## 6. S2 $\rightarrow$ S5 Retry Transition Analysis

Source file: [`results/phase8/table7_retry_transition_matrix.csv`](file:///d:/Indic-QA-RAG/results/phase8/table7_retry_transition_matrix.csv)

A complete paired query accounting was conducted for the 75 answerable queries on **Perturbed Dev** comparing S2 (Verifier only) and S5 (Verifier + Arbiter Retry):

| Transition Category | Initial State in S2 | Final Outcome in S5 | Query Count | Percentage of $N_{\text{ans}}$ | Query IDs |
|---|---|---|---|---|---|
| **Retained Correct** | Correct ($c_{\text{ans}}=63$) | Correct ($c_{\text{ans}}=63$) | **63** | **84.00%** | All 63 correctly answered queries |
| **Retained Refusal** | False Refusal ($r_{\text{ans}}=12$) | False Refusal ($r_{\text{ans}}=7$) | **7** | **9.33%** | 7 retrieval misses with zero gold evidence |
| **Retried $\rightarrow$ Incorrect** | False Refusal ($r_{\text{ans}}=12$) | Incorrect Answer ($w_{\text{ans}}=5$) | **5** | **6.67%** | `5726431aec44d21400f3dd17`, `5730a951069b531400832215`, `5727c55bff5b5019007d94ce`, `5727e21e4b864d1900163f36`, `57263c78ec44d21400f3dc7c` |
| **Total Answerable Queries** | — | — | **75** | **100.00%** | Exact identity holds ($63 + 7 + 5 = 75$) |

### Reconciliation of the Metric Equivalence:
- **Avoided Refusal $\neq$ Increased Correctness:** In S5, retries successfully prevented 5 false refusals ($\text{FRR}$ dropped from $16.00\%$ to $9.33\%$, $-6.67\text{ pp}$).
- **Outcome Destination:** Because round-2 retrieval in Phase 6 did not recover gold evidence for these 5 specific queries, the resulting non-abstaining answers were ungrounded and evaluated as incorrect ($w_{\text{ans}}$ rose from $0$ to $5$, $+6.67\text{ pp}$).
- **Mathematical Invariant:**
  - In S2: $\text{Acc}_{\text{ans}} (84.00\%) + \text{FRR} (16.00\%) + \text{Err}_{\text{ans}} (0.00\%) = 100.00\%$
  - In S5: $\text{Acc}_{\text{ans}} (84.00\%) + \text{FRR} (9.33\%) + \text{Err}_{\text{ans}} (6.67\%) = 100.00\%$
  - $\text{Acc}_{\text{ans}}$ remained identical at **$84.00\%$** ($63/75$).
- **Phase 7 Architectural Resolution:** In Phase 7, the explicit schema-backed state propagation fix (`needs_retry: bool`) and targeted regression test (`test_retry_produces_grounded_answer_when_evidence_permits`) proved that when round-2 retrieval successfully fetches gold evidence, the retry loop recovers a grounded, correctly cited answer.

---

## 7. Leave-One-Agent-Out (LOAO) Component Ablation

Source file: [`results/phase8/table5_loao_ablation.csv`](file:///d:/Indic-QA-RAG/results/phase8/table5_loao_ablation.csv)

Ablating individual agents from S5 on Perturbed Dev ($N=150$):

| Ablation Variant | Removed Agent | $\text{Acc}_{\text{ans}}$ | $\text{FRR}$ | $\text{Err}_{\text{ans}}$ | $\text{Acc}_{\text{unans}}$ | Token F1 | Functional Role Isolated |
|---|---|---|---|---|---|---|---|
| **Full S5 System** | None (All Active) | 84.00% | 9.33% | 6.67% | 100.0% | 0.6777 | Complete multi-agent pipeline |
| **w/o Query Agent** | `QueryAgent` | 84.00% | **16.00%** | **0.00%** | 100.0% | 0.6777 | Removes retry query reformulation; returns to S2 boundary |
| **w/o Evidence Grader** | `EvidenceGrader` | 84.00% | 9.33% | 6.67% | 100.0% | 0.6777 | Disables passage score filtering; Answerer receives raw top-5 |
| **w/o Verifier** | `VerifierAgent` | 84.00% | 9.33% | 6.67% | 100.0% | 0.6777 | Disables claim verification gate |
| **w/o Arbiter Retry** | `AbstentionArbiter` retry | 84.00% | **16.00%** | **0.00%** | 100.0% | 0.6777 | Disables retry loops; abstains immediately upon verification failure |

### Takeaway for Paper Author:
The Query Agent and Abstention Arbiter retry loop act as a coupled unit. Disabling either immediately increases the False Refusal Rate from $9.33\%$ back to $16.00\%$ ($+6.67\text{ pp}$), confirming their specific role in refusal reduction.

---

## 8. Summary of Limitations for Paper Discussion

1. **Test Set Retries Inactive:** Because all 7 false refusals on the held-out Test set were complete retrieval misses (the Answerer explicitly refused, which Arbiter does not retry), zero retries were triggered on Test ($c_{\text{ans}}=31, r_{\text{ans}}=7, w_{\text{ans}}=0$).
2. **Sample Size Constraints:** With $N=150$ (Dev) and $N=75$ (Test), discordant pair counts ($b+c \le 3$) have insufficient statistical power to detect subtle effect sizes ($\le 3\%$) at $p < 0.05$.
3. **Hinglish Cross-Script Gap:** Heuristic transliteration fails to close the script gap; neural sequence models or native dual-script indexing are required.
4. **Citation Extraction:** Baseline regex citation extraction achieves $0.0\%$ precision on Llama-3.1 output without fine-tuned citation formatting.
