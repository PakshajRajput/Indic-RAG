"""Statistical analysis module for PJ-09 IndicRAG-QA (Phase 6).

Implements:
- 95% Bootstrap Confidence Intervals for primary metrics (Acc_ans, Acc_unans, Acc_overall, FRR, HR)
- Paired bootstrap difference distributions (System A vs System B)
- McNemar's exact test for paired binary outcome comparison
- Effect size calculations
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np


def compute_bootstrap_ci(
    values: list[float] | np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean and percentile bootstrap confidence interval.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    resampled_means = np.mean(arr[indices], axis=1)

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(resampled_means, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(resampled_means, 100 * (1.0 - alpha / 2.0)))
    mean_val = float(np.mean(arr))

    return mean_val, ci_lower, ci_upper


def compute_paired_bootstrap_difference(
    values_a: list[float] | np.ndarray,
    values_b: list[float] | np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Compute paired bootstrap difference (System A - System B).

    Returns:
        (diff_mean, ci_lower, ci_upper, p_value_two_tailed)
    """
    arr_a = np.asarray(values_a, dtype=float)
    arr_b = np.asarray(values_b, dtype=float)
    if len(arr_a) != len(arr_b) or len(arr_a) == 0:
        return 0.0, 0.0, 0.0, 1.0

    diffs = arr_a - arr_b
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    resampled_diff_means = np.mean(diffs[indices], axis=1)

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(resampled_diff_means, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(resampled_diff_means, 100 * (1.0 - alpha / 2.0)))

    # Empirical p-value under null hypothesis (diff = 0)
    # Centered distribution
    centered_means = resampled_diff_means - mean_diff
    p_val = float(np.mean(np.abs(centered_means) >= np.abs(mean_diff)))

    return mean_diff, ci_lower, ci_upper, p_val


def mcnemar_exact_test(
    outcomes_a: list[bool] | list[int],
    outcomes_b: list[bool] | list[int],
) -> dict[str, Any]:
    """Perform exact McNemar's test for paired binary classification outcomes.

    Contingency table:
                System B (1)  System B (0)
    System A (1)      a             b
    System A (0)      c             d

    b: System A correct, System B incorrect
    c: System A incorrect, System B correct
    """
    if len(outcomes_a) != len(outcomes_b) or len(outcomes_a) == 0:
        return {"b": 0, "c": 0, "total_discordant": 0, "p_value": 1.0, "statistic": 0.0, "odds_ratio": 1.0}

    b = 0
    c = 0
    for y_a, y_b in zip(outcomes_a, outcomes_b, strict=True):
        if y_a and not y_b:
            b += 1
        elif not y_a and y_b:
            c += 1

    total_discordant = b + c
    if total_discordant == 0:
        return {"b": b, "c": c, "total_discordant": 0, "p_value": 1.0, "statistic": 0.0, "odds_ratio": 1.0}

    # Chi-square with Edwards continuity correction if total >= 25, else exact binomial
    if total_discordant >= 25:
        statistic = ((abs(b - c) - 1.0) ** 2) / total_discordant
        # Approximation via standard normal survival
        from scipy.stats import chi2
        p_value = float(chi2.sf(statistic, df=1))
    else:
        # Exact two-tailed binomial calculation
        from scipy.stats import binom
        k = min(b, c)
        p_value = float(min(1.0, 2.0 * binom.cdf(k, total_discordant, 0.5)))
        statistic = float(abs(b - c))

    odds_ratio = (b / c) if c > 0 else float("inf") if b > 0 else 1.0

    return {
        "b": b,
        "c": c,
        "total_discordant": total_discordant,
        "statistic": statistic,
        "p_value": p_value,
        "is_significant_05": p_value < 0.05,
        "odds_ratio": odds_ratio,
    }


def analyze_metric_distribution(items: list[dict[str, Any]], metric_key: str) -> dict[str, float]:
    """Calculate mean and 95% bootstrap confidence intervals for a metric across items."""
    vals = [float(it[metric_key]) for it in items if metric_key in it]
    mean, lower, upper = compute_bootstrap_ci(vals)
    return {
        "mean": mean,
        "ci_95_lower": lower,
        "ci_95_upper": upper,
        "sample_size": len(vals),
    }
