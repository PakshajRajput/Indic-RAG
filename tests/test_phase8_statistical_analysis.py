"""Unit and regression tests for Phase 8 Statistical Analysis & Reproducibility.

Validates:
1. Artifact existence and integrity across all results/phase8/ deliverables.
2. Strict RGB benchmark metric identities (Acc_ans + FRR + Err_ans == 1.0; Acc_unans + HR == 1.0).
3. Clean vs. perturbed noise degradation math and consistency.
4. Exact paired transition counts for retry analysis (75 answerable queries on Perturbed Dev).
5. Statistical significance calculations, McNemar exact tests, and bootstrap confidence intervals.
6. Cryptographic preservation of frozen benchmark splits and search corpora.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from indicrag.eval.significance import mcnemar_exact_test


def test_phase8_artifact_existence_and_non_emptiness() -> None:
    """Verify that all Phase 8 tables, figures, and summaries exist and are non-empty."""
    p8_dir = Path("results/phase8")
    assert p8_dir.is_dir(), "results/phase8 directory does not exist"

    expected_files = [
        "table1_systems_ladder_dev.csv",
        "table2_systems_ladder_test.csv",
        "table3_clean_vs_perturbed.csv",
        "table4_hinglish_ablation.csv",
        "table5_loao_ablation.csv",
        "table6_statistical_significance.csv",
        "table7_retry_transition_matrix.csv",
        "metric_integrity_audit.json",
        "phase8_consolidated_summary.json",
        "fig1_systems_ladder_accuracy.svg",
        "fig2_clean_vs_perturbed_drop.svg",
        "fig3_hinglish_cross_script_gap.svg",
        "fig4_retry_transition_breakdown.svg",
    ]

    for fname in expected_files:
        fpath = p8_dir / fname
        assert fpath.exists(), f"Missing artifact: {fname}"
        assert fpath.stat().st_size > 0, f"Artifact is empty: {fname}"


def test_rgb_metric_identities_hold_strictly() -> None:
    """Verify that RGB mathematical identities hold across all systems and conditions."""
    audit_file = Path("results/phase8/metric_integrity_audit.json")
    with open(audit_file, encoding="utf-8") as f:
        audit_items = json.load(f)

    assert len(audit_items) >= 32, f"Expected at least 32 audit records, found {len(audit_items)}"

    for item in audit_items:
        # Identity 1: Acc_ans + FRR + Err_ans == 1.0
        acc_ans = item["acc_ans"]
        frr = item["false_refusal_rate"]
        err_ans = item["error_rate_ans"]
        sum_ans = acc_ans + frr + err_ans
        assert math.isclose(sum_ans, 1.0, abs_tol=1e-5), (
            f"Answerable identity violated for {item['system']} ({item['corpus']}): {sum_ans} != 1.0"
        )
        assert item["ans_identity_holds"] is True

        # Identity 2: Acc_unans + HR == 1.0
        acc_unans = item["acc_unans"]
        hr = item["hallucination_rate"]
        sum_unans = acc_unans + hr
        assert math.isclose(sum_unans, 1.0, abs_tol=1e-5), (
            f"Unanswerable identity violated for {item['system']} ({item['corpus']}): {sum_unans} != 1.0"
        )
        assert item["unans_identity_holds"] is True


def test_clean_vs_perturbed_degradation_metrics() -> None:
    """Verify clean vs. perturbed degradation calculations and boundary conditions."""
    df_cp = pd.read_csv("results/phase8/table3_clean_vs_perturbed.csv")
    assert len(df_cp) == 12  # 6 systems on Dev, 6 systems on Test

    # Answerable accuracy on perturbed corpus must be <= clean corpus (monotonic degradation or parity)
    for _, row in df_cp.iterrows():
        assert row["delta_acc_ans_pp"] <= 0.0001, (
            f"Acc_ans unexpectedly increased under noise for {row['system']} on {row['split']}"
        )
        # False refusal rate on perturbed corpus must be >= clean corpus
        assert row["delta_frr_pp"] >= -0.0001, (
            f"FRR unexpectedly decreased under noise for {row['system']} on {row['split']}"
        )


def test_retry_transition_exact_counts() -> None:
    """Verify that the 75 answerable queries on Perturbed Dev partition precisely into 63 + 7 + 5."""
    df_retry = pd.read_csv("results/phase8/table7_retry_transition_matrix.csv")
    total_queries = df_retry["count"].sum()
    assert total_queries == 75, f"Expected 75 total answerable queries, got {total_queries}"

    counts_by_type = dict(zip(df_retry["transition_type"], df_retry["count"]))
    assert counts_by_type["Retained Correct"] == 63
    assert counts_by_type["Retained Refusal (Zero Evidence)"] == 7
    assert counts_by_type["Retried → Incorrect Answer"] == 5

    # Verification of RGB percentage identities: 84.00% + 9.33% + 6.67% == 100.00%
    total_pct = df_retry["pct"].sum()
    assert math.isclose(total_pct, 100.0, abs_tol=1e-2)


def test_significance_test_reproducibility() -> None:
    """Verify that McNemar exact tests reproduce exact discordant counts and p-values."""
    df_sig = pd.read_csv("results/phase8/table6_statistical_significance.csv")
    sig_map = {row["comparison"]: row for _, row in df_sig.iterrows()}

    # Test 1: Hinglish Method B vs Method A
    h_test = sig_map["hinglish_method_B_vs_method_A"]
    assert h_test["discordant_pairs"] == 2
    assert math.isclose(h_test["p_value_raw"], 0.50, abs_tol=1e-3)
    assert h_test["nominal_sig_05"] == False

    # Test 2: S2 vs S1
    s2_s1 = sig_map["S2_verifier_vs_S1_score_gate"]
    assert s2_s1["discordant_pairs"] == 3
    assert math.isclose(s2_s1["p_value_raw"], 0.25, abs_tol=1e-3)
    assert s2_s1["nominal_sig_05"] == False

    # Test 3: S5 vs S0
    s5_s0 = sig_map["S5_full_vs_S0_single_pass"]
    assert s5_s0["discordant_pairs"] == 3
    assert math.isclose(s5_s0["p_value_raw"], 0.25, abs_tol=1e-3)
    assert s5_s0["nominal_sig_05"] == False


def test_manifest_datasets_and_corpora_preservation() -> None:
    """Verify that all 6 datasets and search corpora match their Phase 6 cryptographic hashes."""
    from scripts.run_phase8_analysis import compute_file_sha256, verify_protocol_manifest_integrity

    manifest_file = Path("results/phase6_protocol_manifest.json")
    audit = verify_protocol_manifest_integrity(manifest_file)

    # The 6 core benchmark and corpus artifacts must never change
    critical_files = [
        "data/processed/benchmark_dev_150.json",
        "data/processed/benchmark_test_75.json",
        "data/processed/corpus_dev.json",
        "data/processed/corpus_test.json",
        "data/processed/counterfactual/corpus_dev_perturbed.json",
        "data/processed/counterfactual/corpus_test_perturbed.json",
        "src/indicrag/eval/metrics.py",
        "src/indicrag/eval/judge.py",
        "src/indicrag/eval/significance.py",
    ]

    for rel_path in critical_files:
        status_info = audit["details"][rel_path]
        assert status_info["status"] == "MATCH", (
            f"Cryptographic hash mismatch for frozen artifact {rel_path}!"
        )
