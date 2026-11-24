"""Statistical tables, effect-direction AUCs, and confidence ellipses."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def build_contingency_table(labels: Iterable, values: Iterable) -> np.ndarray:
    """Build a two-dimensional count table from paired labels and values."""
    label_levels = list(dict.fromkeys(labels))
    value_levels = list(dict.fromkeys(values))
    label_index = {value: index for index, value in enumerate(label_levels)}
    value_index = {value: index for index, value in enumerate(value_levels)}
    table = np.zeros((len(label_levels), len(value_levels)), dtype=np.uint32)
    for label, value in zip(labels, values):
        table[label_index[label], value_index[value]] += 1
    return table


def direction_independent_auc(labels: Iterable, scores: Iterable) -> float:
    """Return the source workflow's direction-adjusted ROC AUC.

    The original implementation recoded the binary label when the direct AUC
    was below 0.5. Algebraically, that is ``1 - AUC`` for non-constant scores;
    using the complement also avoids materializing a second label vector.
    """
    auc = float(roc_auc_score(y_true=list(labels), y_score=list(scores)))
    return 1.0 - auc if auc < 0.5 else auc


def add_fdr(p_values: pd.Series, column_name: str = "fdr") -> pd.Series:
    """Add Benjamini-Hochberg false-discovery rates for a p-value vector."""
    from scipy.stats import false_discovery_control

    return pd.Series(false_discovery_control(p_values), index=p_values.index, name=column_name)
