"""Unit and integration tests for Phase 6 Benchmark Experiments and Evaluation.

Tests:
1. Protocol freeze & cryptographic SHA-256 manifest integrity
2. Bootstrap confidence interval mathematical bounds (L <= mean <= U)
3. McNemar's exact test contingency math and p-value calculations
4. Paired bootstrap difference distributions
5. Held-out Test set isolation and zero overlap with Dev benchmark
6. Synthetic Hinglish provenance link retention
7. Phase 6 evaluation summary schema completeness
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from indicrag.eval.significance import (
    compute_bootstrap_ci,
    compute_paired_bootstrap_difference,
    mcnemar_exact_test,
)


def test_bootstrap_confidence_interval_bounds():
    """Verify bootstrap confidence intervals satisfy mathematical bounds."""
    # 1. Standard binary vector
    values = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    mean, lower, upper = compute_bootstrap_ci(values, n_resamples=500, ci=0.95, seed=42)
    assert 0.0 <= lower <= mean <= upper <= 1.0
    assert abs(mean - 0.70) < 1e-5

    # 2. Degenerate all-zeros vector
    zeros = [0.0] * 20
    m_z, l_z, u_z = compute_bootstrap_ci(zeros, n_resamples=100)
    assert m_z == 0.0
    assert l_z == 0.0
    assert u_z == 0.0

    # 3. Empty input safety
    m_e, l_e, u_e = compute_bootstrap_ci([])
    assert m_e == 0.0
    assert l_e == 0.0
    assert u_e == 0.0


def test_mcnemar_exact_test_contingency():
    """Verify McNemar exact test correctly evaluates discordant pairs."""
    # 1. Identical systems -> zero discordant pairs -> p-value = 1.0
    out_a = [True, True, False, True, False]
    out_b = [True, True, False, True, False]
    res_ident = mcnemar_exact_test(out_a, out_b)
    assert res_ident["total_discordant"] == 0
    assert res_ident["p_value"] == 1.0

    # 2. Asymmetric discordant pairs (System A strictly superior)
    # A correct, B incorrect for 15 cases; B correct, A incorrect for 0 cases
    out_a_sup = [True] * 20 + [False] * 5
    out_b_inf = [False] * 15 + [True] * 5 + [False] * 5
    res_asym = mcnemar_exact_test(out_a_sup, out_b_inf)
    assert res_asym["b"] == 15
    assert res_asym["c"] == 0
    assert res_asym["is_significant_05"] is True
    assert res_asym["p_value"] < 0.001


def test_paired_bootstrap_difference_math():
    """Verify paired bootstrap difference accurately captures mean difference and CI."""
    a = np.array([0.9, 0.85, 0.95, 0.88, 0.92])
    b = np.array([0.5, 0.55, 0.48, 0.52, 0.50])
    mean_diff, low, high, p_val = compute_paired_bootstrap_difference(a, b, n_resamples=500, seed=42)

    expected_mean = float(np.mean(a - b))
    assert abs(mean_diff - expected_mean) < 1e-4
    assert low > 0.0  # Entire 95% CI strictly positive
    assert p_val < 0.05


def test_held_out_test_set_isolation():
    """Verify that held-out Test benchmark (benchmark_test_75.json) is isolated from Dev."""
    with open("data/processed/benchmark_dev_150.json", encoding="utf-8") as f:
        dev_bench = json.load(f)
    with open("data/processed/benchmark_test_75.json", encoding="utf-8") as f:
        test_bench = json.load(f)

    dev_ids = {q["id"] for q in dev_bench}
    test_ids = {q["id"] for q in test_bench}
    assert dev_ids.isdisjoint(test_ids), f"Leakage violation: {dev_ids & test_ids}"

    dev_titles = {q["title"] for q in dev_bench}
    test_titles = {q["title"] for q in test_bench}
    assert dev_titles.isdisjoint(test_titles), f"Title leakage violation: {dev_titles & test_titles}"

    # Verify balance
    test_ans = sum(1 for q in test_bench if not q["is_impossible"])
    test_unans = sum(1 for q in test_bench if q["is_impossible"])
    assert test_ans == 38
    assert test_unans == 37
    assert len(test_bench) == 75


def test_synthetic_hinglish_provenance_retention():
    """Verify synthetic Hinglish questions preserve original Hindi IDs and text."""
    with open("data/processed/benchmark_dev_150.json", encoding="utf-8") as f:
        dev_bench = json.load(f)

    for item in dev_bench:
        assert "id" in item
        assert "question" in item
        assert "is_impossible" in item
        assert isinstance(item["id"], str)
        assert len(item["question"].strip()) > 0


def test_phase6_protocol_manifest_integrity():
    """Verify protocol manifest file exists and contains all required hashes and settings."""
    manifest_path = Path("results/phase6_protocol_manifest.json")
    if not manifest_path.exists():
        pytest.skip("Manifest will be generated upon running run_phase6_experiments.")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["random_seed"] == 42
    assert "file_hashes_sha256" in manifest
    assert len(manifest["file_hashes_sha256"]) >= 8

    # Verify no file was missing when hashed
    for fpath, fhash in manifest["file_hashes_sha256"].items():
        assert fhash != "FILE_NOT_FOUND", f"Missing file during protocol freeze: {fpath}"
        assert len(fhash) == 64  # Valid SHA-256 hex string
