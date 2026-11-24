"""Threshold selection and binary evaluation without label flipping."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_ORDER = ("AUROC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1")


def bootstrap_auc_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> tuple[float, float, float]:
    """Return the AUC point estimate and a percentile bootstrap interval."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    point_estimate = roc_auc_without_flipping(labels, scores)
    if np.unique(labels).size < 2 or n_bootstrap <= 0:
        return point_estimate, np.nan, np.nan

    rng = np.random.default_rng(random_state)
    bootstrapped: list[float] = []
    size = len(labels)
    for _ in range(n_bootstrap):
        index = rng.integers(0, size, size)
        if np.unique(labels[index]).size < 2:
            continue
        bootstrapped.append(float(roc_auc_score(labels[index], scores[index])))
    if not bootstrapped:
        return point_estimate, np.nan, np.nan
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(bootstrapped, [alpha, 1 - alpha])
    return point_estimate, float(lower), float(upper)


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return (np.nan, np.nan)
    from scipy.stats import norm

    z = float(norm.ppf(1 - (1 - confidence) / 2))
    denominator = 1 + z**2 / trials
    centre = (successes / trials + z**2 / (2 * trials)) / denominator
    half_width = z * np.sqrt(successes / trials * (1 - successes / trials) / trials + z**2 / (4 * trials**2)) / denominator
    return float(centre - half_width), float(centre + half_width)


def roc_auc_without_flipping(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC and return NaN for a single-class evaluation set."""
    labels = np.asarray(labels)
    if np.unique(labels).size < 2:
        return np.nan
    return float(roc_auc_score(labels, scores))


def find_sensitivity_cutoff(
    prediction_frame: pd.DataFrame,
    min_sensitivity: float = 0.95,
    steps: int = 100,
) -> float:
    """Find the highest threshold whose sensitivity remains at least the target.

    This reproduces the source threshold-search order while removing the
    prohibited automatic label-flipping behavior.
    """
    require_columns(prediction_frame, ["label", "pred"])
    labels = prediction_frame["label"].to_numpy(dtype=int)
    scores = prediction_frame["pred"].to_numpy(dtype=float)

    selected = -1.0
    for threshold in np.linspace(0, 1, steps + 1):
        predicted = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        if sensitivity < min_sensitivity:
            break
        selected = float(threshold)
    return float(selected)


def evaluate_predictions(
    prediction_frame: pd.DataFrame,
    cutoff: float,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compute binary point estimates and Wilson intervals at a fixed cutoff."""
    require_columns(prediction_frame, ["label", "pred"])
    labels = prediction_frame["label"].to_numpy(dtype=int)
    scores = prediction_frame["pred"].to_numpy(dtype=float)
    predicted = (scores >= cutoff).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    auroc = roc_auc_without_flipping(labels, scores)
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = precision_score(labels, predicted, zero_division=0)
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = f1_score(labels, predicted, zero_division=0)
    accuracy = accuracy_score(labels, predicted)

    intervals = {
        "Sensitivity": wilson_score_interval(tp, tp + fn, confidence),
        "Specificity": wilson_score_interval(tn, tn + fp, confidence),
        "PPV": wilson_score_interval(tp, tp + fp, confidence),
        "NPV": wilson_score_interval(tn, tn + fn, confidence),
        "Accuracy": wilson_score_interval(tp + tn, tp + tn + fp + fn, confidence),
        "F1": (np.nan, np.nan),
        "AUROC": (np.nan, np.nan),
    }

    if np.unique(labels).size >= 2:
        fpr, tpr, thresholds = roc_curve(labels, scores)
    else:
        fpr = tpr = thresholds = np.array([])

    return {
        "AUROC": auroc,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "F1": f1,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "intervals": intervals,
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds},
    }


def evaluation_summary(
    result: dict[str, Any],
    model: str,
    dataset: str,
    cutoff: float | None = None,
) -> pd.DataFrame:
    """Convert an evaluation result into a one-row summary table.

    When ``cutoff`` is supplied (the sensitivity-anchored operating threshold
    the metrics were computed at), it is included as a ``cutoff`` column so
    downstream users can read the operating point directly from the summary.
    """
    row: dict[str, Any] = {"model": model, "dataset": dataset}
    if cutoff is not None:
        row["cutoff"] = cutoff
    for metric in METRIC_ORDER:
        value = result[metric]
        lower, upper = result["intervals"][metric]
        row[metric] = value
        row[f"{metric}_lower"] = lower
        row[f"{metric}_upper"] = upper
    row.update({"TP": result["TP"], "TN": result["TN"], "FP": result["FP"], "FN": result["FN"]})
    return pd.DataFrame([row])


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error when required prediction columns are absent."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def precision_recall_with_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> dict[str, Any]:
    """Compute the PR curve, average precision, and a bootstrap AP interval.

    Returns a dict with ``ap``, ``ap_lower``, ``ap_upper``, ``precision``,
    ``recall``, and ``thresholds`` arrays. Bootstrap resamples with a single
    observed class are skipped.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(random_state)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    ap_point = float(average_precision_score(labels, scores))

    size = len(labels)
    alpha = (1 - confidence) / 2
    bootstrapped: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, size, size)
        if np.unique(labels[indices]).size < 2:
            continue
        bootstrapped.append(float(average_precision_score(labels[indices], scores[indices])))

    if bootstrapped:
        ap_lower, ap_upper = (float(v) for v in np.quantile(bootstrapped, [alpha, 1 - alpha]))
    else:
        ap_lower, ap_upper = np.nan, np.nan

    return {
        "ap": ap_point,
        "ap_lower": ap_lower,
        "ap_upper": ap_upper,
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
    }


def delong_test(
    labels: np.ndarray,
    scores_first: np.ndarray,
    scores_second: np.ndarray,
) -> dict[str, Any]:
    """Compare two correlated ROC AUCs with the DeLong test.

    The implementation follows the standard formulation: per-sample
    placement values (V10 for positives, V01 for negatives) are computed for
    both models, their empirical covariances give the AUC variances and the
    AUC covariance, and the difference is tested against a standard normal.

    Parameters
    ----------
    labels : array-like
        Binary true labels (0/1).
    scores_first, scores_second : array-like
        Prediction scores of the two models on the same participants.

    Returns
    -------
    dict
        ``auc_first``, ``auc_second``, ``auc_difference``, ``z_statistic``,
        ``p_value``, ``variance_difference``, ``covariance`` and the class
        counts.
    """
    from scipy import stats as scipy_stats

    labels = np.asarray(labels)
    scores_first = np.asarray(scores_first, dtype=float)
    scores_second = np.asarray(scores_second, dtype=float)

    auc_first = float(roc_auc_score(labels, scores_first))
    auc_second = float(roc_auc_score(labels, scores_second))
    auc_difference = auc_first - auc_second

    positive_mask = labels == 1
    n_positive = int(np.sum(positive_mask))
    n_negative = int(np.sum(~positive_mask))
    positive_scores_first = scores_first[positive_mask]
    negative_scores_first = scores_first[~positive_mask]
    positive_scores_second = scores_second[positive_mask]
    negative_scores_second = scores_second[~positive_mask]

    def placement_values(positive_scores: np.ndarray, negative_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-positive and per-negative placement values (with tie handling)."""
        v10 = np.zeros(len(positive_scores))
        for i, score in enumerate(positive_scores):
            v10[i] = np.sum(negative_scores < score) + 0.5 * np.sum(negative_scores == score)
        v10 /= len(negative_scores)

        v01 = np.zeros(len(negative_scores))
        for i, score in enumerate(negative_scores):
            v01[i] = np.sum(positive_scores > score) + 0.5 * np.sum(positive_scores == score)
        v01 /= len(positive_scores)
        return v10, v01

    v10_first, v01_first = placement_values(positive_scores_first, negative_scores_first)
    v10_second, v01_second = placement_values(positive_scores_second, negative_scores_second)

    covariance_positive = (
        np.sum((v10_first - auc_first) * (v10_second - auc_second)) / (n_positive - 1)
        if n_positive > 1
        else 0.0
    )
    covariance_negative = (
        np.sum((v01_first - auc_first) * (v01_second - auc_second)) / (n_negative - 1)
        if n_negative > 1
        else 0.0
    )

    var_positive_first = float(np.var(v10_first, ddof=1)) if n_positive > 1 else 0.0
    var_negative_first = float(np.var(v01_first, ddof=1)) if n_negative > 1 else 0.0
    var_positive_second = float(np.var(v10_second, ddof=1)) if n_positive > 1 else 0.0
    var_negative_second = float(np.var(v01_second, ddof=1)) if n_negative > 1 else 0.0

    var_first = var_positive_first / n_positive + var_negative_first / n_negative
    var_second = var_positive_second / n_positive + var_negative_second / n_negative
    covariance = covariance_positive / n_positive + covariance_negative / n_negative

    variance_difference = var_first + var_second - 2 * covariance
    variance_difference = max(variance_difference, 1e-10)  # guard against numeric error

    z_statistic = float(auc_difference / np.sqrt(variance_difference))
    p_value = float(2 * (1 - scipy_stats.norm.cdf(abs(z_statistic))))

    return {
        "auc_first": auc_first,
        "auc_second": auc_second,
        "auc_difference": auc_difference,
        "z_statistic": z_statistic,
        "p_value": p_value,
        "variance_difference": float(variance_difference),
        "covariance": float(covariance),
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def adjust_ppv_npv(
    prevalence: float,
    sensitivity: float,
    specificity: float,
    n_sensitivity: Optional[int] = None,
    n_specificity: Optional[int] = None,
    n_prevalence: Optional[int] = None,
    method: str = "bootstrap",
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> dict[str, Any]:
    """Adjust PPV and NPV for a target-population prevalence.

    Point estimates follow Bayes' theorem from the sensitivity, specificity,
    and prevalence. Confidence intervals can be computed by sampling
    Beta posteriors for sensitivity/specificity (``bootstrap`` and
    ``bayesian``) or by a delta-method Wilson approximation (``wilson``).

    Parameters
    ----------
    prevalence : float
        Target-population prevalence in (0, 1).
    sensitivity, specificity : float
        Diagnostic sensitivity and specificity in (0, 1).
    n_sensitivity, n_specificity : int, optional
        Sample sizes behind the sensitivity/specificity estimates; used to
        build Beta sampling distributions (Laplace-smoothed).
    n_prevalence : int, optional
        Sample size behind the prevalence estimate; adds sampling noise.
    method : "bootstrap", "wilson", or "bayesian"
        Interval method. ``bootstrap`` and ``bayesian`` differ only in the
        prevalence handling (fixed versus Beta-sampled).
    n_bootstrap : int
        Number of Monte Carlo draws.
    confidence : float
        Confidence level.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    dict
        Point estimates, interval bounds, likelihood ratios, and the method
        metadata.
    """
    from scipy.stats import beta as beta_distribution

    p = float(prevalence)
    se = float(sensitivity)
    sp = float(specificity)

    tp_rate = p * se
    fp_rate = (1 - p) * (1 - sp)
    tn_rate = (1 - p) * sp
    fn_rate = p * (1 - se)

    ppv_point = tp_rate / (tp_rate + fp_rate) if (tp_rate + fp_rate) > 0 else np.nan
    npv_point = tn_rate / (tn_rate + fn_rate) if (tn_rate + fn_rate) > 0 else np.nan

    alpha = (1 - confidence) / 2
    rng = np.random.default_rng(random_state)

    if method in ("bootstrap", "bayesian"):
        if n_sensitivity and n_specificity:
            # Beta(alpha, beta) with Laplace smoothing from the counts.
            se_samples = beta_distribution.rvs(se * n_sensitivity + 1, (1 - se) * n_sensitivity + 1, size=n_bootstrap, random_state=rng)
            sp_samples = beta_distribution.rvs(sp * n_specificity + 1, (1 - sp) * n_specificity + 1, size=n_bootstrap, random_state=rng)
        else:
            # Without sample sizes, assume a 2.5% measurement error.
            se_samples = np.clip(se + rng.normal(0, 0.025, n_bootstrap), 0.001, 0.999)
            sp_samples = np.clip(sp + rng.normal(0, 0.025, n_bootstrap), 0.001, 0.999)

        if method == "bayesian" and n_prevalence:
            p_samples = beta_distribution.rvs(p * n_prevalence + 1, (1 - p) * n_prevalence + 1, size=n_bootstrap, random_state=rng)
        elif n_prevalence:
            p_samples = np.clip(p + rng.normal(0, np.sqrt(p * (1 - p) / n_prevalence), n_bootstrap), 0.001, 0.999)
        else:
            p_samples = np.full(n_bootstrap, p)

        ppv_samples = (p_samples * se_samples) / (p_samples * se_samples + (1 - p_samples) * (1 - sp_samples))
        npv_samples = ((1 - p_samples) * sp_samples) / ((1 - p_samples) * sp_samples + p_samples * (1 - se_samples))

        ppv_lower, ppv_upper = (float(v) for v in np.nanpercentile(ppv_samples, [alpha * 100, (1 - alpha) * 100]))
        npv_lower, npv_upper = (float(v) for v in np.nanpercentile(npv_samples, [alpha * 100, (1 - alpha) * 100]))
    elif method == "wilson":
        from scipy.stats import norm

        z = float(norm.ppf(1 - alpha))
        var_se = se * (1 - se) / n_sensitivity if n_sensitivity else 0.02**2
        var_sp = sp * (1 - sp) / n_specificity if n_specificity else 0.02**2

        # Delta-method approximation of the PPV variance.
        denom_ppv = p * se + (1 - p) * (1 - sp)
        d_ppv_d_se = p * (1 - p) * (1 - sp) / denom_ppv**2
        d_ppv_d_sp = p * se * (1 - p) / denom_ppv**2
        var_ppv = d_ppv_d_se**2 * var_se + d_ppv_d_sp**2 * var_sp
        ppv_se = np.sqrt(var_ppv)
        ppv_lower = float(min(1.0, max(0.0, ppv_point - z * ppv_se)))
        ppv_upper = float(min(1.0, max(0.0, ppv_point + z * ppv_se)))

        denom_npv = (1 - p) * sp + p * (1 - se)
        d_npv_d_se = p * (1 - p) * sp / denom_npv**2
        d_npv_d_sp = (1 - p) * p * (1 - se) / denom_npv**2
        var_npv = d_npv_d_se**2 * var_se + d_npv_d_sp**2 * var_sp
        npv_se = np.sqrt(var_npv)
        npv_lower = float(min(1.0, max(0.0, npv_point - z * npv_se)))
        npv_upper = float(min(1.0, max(0.0, npv_point + z * npv_se)))
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "prevalence": p,
        "sensitivity": se,
        "specificity": sp,
        "ppv_point": ppv_point,
        "ppv_lower": ppv_lower,
        "ppv_upper": ppv_upper,
        "npv_point": npv_point,
        "npv_lower": npv_lower,
        "npv_upper": npv_upper,
        "lr_positive": se / (1 - sp) if (1 - sp) > 0 else np.inf,
        "lr_negative": (1 - se) / sp if sp > 0 else np.inf,
        "method": method,
        "confidence": confidence,
    }
