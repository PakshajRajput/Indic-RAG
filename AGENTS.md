# AGENTS.md — rules for every agent working in this repo

## Mission
Build and evaluate a MULTI-AGENT, abstention-aware RAG system and benchmark for Hindi (Devanagari) and Hinglish (Roman script) QA, following the evaluation protocol of RGB (Chen et al., AAAI 2024). Full brief: docs/MASTER_BRIEF.md.

## Non-negotiable rules
1. NEVER fabricate results, citations, dataset fields, model ids, or library APIs. If you have not run it or opened the source, say "unverified".
2. Every number in README, docs, or paper drafts must come from a file in results/ and be reproducible by a script. Cite the file path next to the number.
3. If a verification step fails, STOP that item, report it in docs/VERIFICATION_REPORT.md, propose 2 fallbacks, and wait. Do not silently substitute another dataset, model, or metric.
4. Work one phase at a time. At the end of each phase: run tests, write a walkthrough with metrics, list open issues, and STOP.
5. Fixed random seeds (seed=42 default). Split train/dev/test by ARTICLE (title), never by question, to prevent leakage. Add a test that fails on leakage.
6. Config-driven: models, k, noise rates, thresholds live in configs/*.yaml. No magic numbers in code.
7. Secrets only in .env (git-ignored). Never print or commit keys. Never send dataset or code to services other than the ones named in the brief.
8. Licensing: IndicSQuAD is CC-BY-4.0 (attribute it). RGB code/data is CC-BY-NC-SA-4.0 — do NOT copy code or data; re-implement the protocol from the paper and cite it. Do not copy code or data from third-party repos unless their license allows it.
9. Prefer small, tested modules (pytest). Type hints, docstrings, ruff-clean.
10. Cost control: cache every model call (disk cache keyed by model+prompt+params). Estimate GPU hours and API calls before big runs and report the estimate.
11. When writing about results, state hypotheses as hypotheses. Do not claim "improves" unless a paired significance test or bootstrap CI supports it.
12. Multi-agent discipline: agents exchange only typed state (Pydantic), each has its own prompt file and model setting, every agent is stub-testable, and the Supervisor enforces a hard per-question budget. Do not add agents beyond the brief without approval.
13. Every multi-agent claim needs the ablation behind it (systems ladder, leave-one-agent-out) and the cost next to it. Report negative or mixed results as they are.
14. The Judge agent is evaluation-only and must use a different model family than the answerer; never use it inside the serving pipeline.

## Environment
Python 3.10.6 (Windows). CPU-only machine (no CUDA). All installs go into D:\Indic-QA-RAG\venv\ (project venv) or D:\Indic-QA-RAG\venv_xlit\ (IndicXlit isolated venv). Do NOT modify the system Python environment.
Generation requires hosted API (Google AI Studio / Together.ai / Groq) — no local large model inference.
Pin versions in requirements/*.txt after Phase 0 succeeds.
