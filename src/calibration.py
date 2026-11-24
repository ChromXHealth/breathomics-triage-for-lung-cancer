"""Calibration analysis: calibration curves and Brier score.

Vendored from the internal analysis library and translated to English. The
binning, bootstrap interval, and Brier decomposition logic is preserved
exactly so the published workflow reproduces the manuscript numbers.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.utils import resample

__all__ = ["compute_calibration_curve", "compute_brier_score"]


def compute_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
    return_ci: bool = True,
    n_bootstrap: int = 1000,
    ci_alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute a calibration curve with per-bin bootstrap intervals.

    Parameters
    ----------
    y_true : array-like
        Binary true labels (0/1).
    y_pred : array-like
        Predicted probabilities in [0, 1].
    n_bins : int
        Number of bins (10 is the clinical standard).
    strategy : "uniform" or "quantile"
        Uniform-width bins (clinical standard) or equal-count bins.
    return_ci : bool
        Whether to compute bootstrap confidence intervals per bin.
    n_bootstrap : int
        Number of bootstrap resamples per bin.
    ci_alpha : float
        Significance level (0.05 gives a 95% CI).

    Returns
    -------
    dict
        Keys ``bin_centers``, ``fraction_positives``, ``counts``,
        ``std_errors``, ``n_bins``, ``strategy`` and, when ``return_ci``,
        ``ci_lower`` / ``ci_upper`` (NaN for bins with too few samples).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if strategy == "uniform":
        # Uniform-width bins: 0-0.1, 0.1-0.2, ... (clinical standard).
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    else:  # quantile
        # Equal-count bins.
        quantiles = np.linspace(0, 1, n_bins + 1) * 100
        bin_boundaries = np.percentile(y_pred, quantiles)
        bin_boundaries[-1] = 1.0  # ensure the maximum value is included
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_centers: list[float] = []
    fraction_positives: list[float] = []
    counts: list[int] = []
    std_errors: list[float] = []

    for lower, upper in zip(bin_lowers, bin_uppers):
        # Left-closed / right-open intervals; the last bin is closed.
        if upper == 1.0:
            in_bin = (y_pred >= lower) & (y_pred <= upper)
        else:
            in_bin = (y_pred >= lower) & (y_pred < upper)

        n_in_bin = int(np.sum(in_bin))
        counts.append(n_in_bin)

        if n_in_bin > 0:
            bin_centers.append(float((lower + upper) / 2))
            fraction_positives.append(float(np.mean(y_true[in_bin])))
            std_errors.append(
                float(np.sqrt(np.mean(y_true[in_bin]) * (1 - np.mean(y_true[in_bin])) / n_in_bin))
                if n_in_bin > 1
                else 0.0
            )
        else:
            # Empty bin: use the interval midpoint and a NaN fraction.
            bin_centers.append(float((lower + upper) / 2))
            fraction_positives.append(np.nan)
            std_errors.append(np.nan)

    result: dict[str, Any] = {
        "bin_centers": np.array(bin_centers),
        "fraction_positives": np.array(fraction_positives),
        "counts": np.array(counts),
        "std_errors": np.array(std_errors),
        "n_bins": n_bins,
        "strategy": strategy,
    }

    # Bootstrap confidence intervals (required for clinical reporting).
    if return_ci:
        ci_lower: list[float] = []
        ci_upper: list[float] = []

        for lower, upper in zip(bin_lowers, bin_uppers):
            if upper == 1.0:
                in_bin = (y_pred >= lower) & (y_pred <= upper)
            else:
                in_bin = (y_pred >= lower) & (y_pred < upper)

            n_in_bin = int(np.sum(in_bin))

            if n_in_bin > 5:  # only compute a CI when the bin is large enough
                boot_means = []
                y_true_bin = y_true[in_bin]
                for _ in range(n_bootstrap):
                    boot_sample = resample(y_true_bin, random_state=None)
                    boot_means.append(np.mean(boot_sample))

                boot_means = np.sort(boot_means)
                lower_idx = int(n_bootstrap * ci_alpha / 2)
                upper_idx = int(n_bootstrap * (1 - ci_alpha / 2))
                ci_lower.append(float(boot_means[lower_idx]))
                ci_upper.append(float(boot_means[upper_idx]))
            else:
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)

        result["ci_lower"] = np.array(ci_lower)
        result["ci_upper"] = np.array(ci_upper)

    return result


def compute_brier_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    return_ci: bool = True,
    n_bootstrap: int = 1000,
    ci_alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute the Brier score with its decomposition and bootstrap interval.

    Decomposition (Murphy, 1973):
    ``Brier = Reliability - Resolution + Uncertainty`` where reliability
    measures calibration error (lower is better) and resolution measures
    discrimination ability (higher is better).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    brier = float(brier_score_loss(y_true, y_pred))

    result: dict[str, Any] = {
        "brier_score": brier,
        "n_samples": int(len(y_true)),
    }

    # Brier score decomposition for clinical interpretation.
    mean_observed = float(np.mean(y_true))
    uncertainty = mean_observed * (1 - mean_observed)

    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        if i == n_bins - 1:  # the last bin includes 1.0
            in_bin = (y_pred >= lower) & (y_pred <= upper)
        else:
            in_bin = (y_pred >= lower) & (y_pred < upper)

        if np.sum(in_bin) > 0:
            avg_pred_in_bin = float(np.mean(y_pred[in_bin]))
            avg_true_in_bin = float(np.mean(y_true[in_bin]))
            n_in_bin = int(np.sum(in_bin))

            reliability += (n_in_bin / len(y_true)) * (avg_pred_in_bin - avg_true_in_bin) ** 2
            resolution += (n_in_bin / len(y_true)) * (avg_true_in_bin - mean_observed) ** 2

    result["reliability"] = reliability
    result["resolution"] = resolution
    result["uncertainty"] = uncertainty

    # Bootstrap confidence interval for the Brier score.
    if return_ci:
        boot_scores = []
        for _ in range(n_bootstrap):
            indices = resample(range(len(y_true)), random_state=None)
            boot_scores.append(brier_score_loss(y_true[indices], y_pred[indices]))

        boot_scores = np.sort(boot_scores)
        lower_idx = int(n_bootstrap * ci_alpha / 2)
        upper_idx = int(n_bootstrap * (1 - ci_alpha / 2))
        result["brier_ci_lower"] = float(boot_scores[lower_idx])
        result["brier_ci_upper"] = float(boot_scores[upper_idx])

    return result
