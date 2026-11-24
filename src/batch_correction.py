"""Batch assessment and correction."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import scale

from .evaluation import require_columns

METADATA_COLUMNS = ["SampleID", "DeviceID", "AnalysisDate", "AnalysisTime", "label"]

TIME_CONDITION_OPERATORS = {
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}


def common_numeric_features(frames: Sequence[pd.DataFrame]) -> list[str]:
    """Return the numerically ordered intersection of features after metadata."""
    if not frames:
        raise ValueError("At least one input table is required.")
    feature_sets = [set(frame.columns[5:]) for frame in frames]
    features = set.intersection(*feature_sets)
    return sorted(features, key=lambda value: int(value))


def prepare_analysis_matrix(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Order features numerically and coerce AnalysisTime for batch assignment."""
    require_columns(frame, METADATA_COLUMNS)
    features = common_numeric_features([frame])
    matrix = frame[frame.columns[:5].tolist() + features].copy()
    matrix["AnalysisTime"] = pd.to_datetime(matrix["AnalysisTime"])
    return matrix, features


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply log10(x + 1) followed by feature-wise standardization."""
    normalized = frame.copy()
    feature_columns = normalized.columns[5:]
    transformed = scale(np.log10(normalized[feature_columns].astype(float) + 1))
    normalized[feature_columns] = transformed
    return normalized


def load_batch_rules(path: Path) -> dict[str, dict]:
    """Read the ``batches`` rule block from a batch-assignment config file."""
    with open(path) as handle:
        rules = yaml.safe_load(handle)["batches"]
    for device_id, device_rules in rules.items():
        if "default" not in device_rules:
            raise ValueError(f"Batch rules for {device_id!r} must define a 'default' batch.")
        for rule in device_rules.get("rules", []):
            unknown = set(rule) - {"batch", *TIME_CONDITION_OPERATORS}
            if unknown:
                raise ValueError(
                    f"Unknown condition(s) {sorted(unknown)} in batch rule {rule!r} "
                    f"for device {device_id!r}."
                )
    return rules


def assign_batch(device_id: str, analysis_time: pd.Timestamp, rules: dict[str, dict]) -> str:
    """Assign a batch label from device- and acquisition-time rules.

    Rules for the sample's DeviceID are evaluated first-match-wins; devices
    without explicit rules use the ``"*"`` fallback block. Samples matching no
    rule receive the device block's ``default`` batch.
    """
    device_rules = rules.get(device_id) or rules.get("*")
    if device_rules is None:
        raise ValueError(
            f"No batch rules for device {device_id!r} and no '*' fallback in the "
            "batch config."
        )
    time = pd.to_datetime(analysis_time)
    for rule in device_rules.get("rules", []):
        conditions = {
            key: pd.Timestamp(value)
            for key, value in rule.items()
            if key in TIME_CONDITION_OPERATORS
        }
        if all(TIME_CONDITION_OPERATORS[key](time, value) for key, value in conditions.items()):
            return rule["batch"]
    return device_rules["default"]


def apply_combat(frame: pd.DataFrame, rules: dict[str, dict]) -> pd.DataFrame:
    """Apply pyComBat to normalized features using the assigned batch labels."""
    require_columns(frame, METADATA_COLUMNS)
    try:
        from combat.pycombat import pycombat
    except ImportError as error:
        raise ImportError("The 'combat' package is required for batch correction.") from error

    combat_input = frame.copy().drop(["AnalysisDate"], axis=1)
    combat_input.insert(
        4,
        "batch_label",
        combat_input.apply(
            lambda row: assign_batch(row["DeviceID"], row["AnalysisTime"], rules), axis=1
        ),
    )
    matrix = combat_input.iloc[:, 5:].copy()
    matrix.index = combat_input["SampleID"]
    corrected = pycombat(matrix.T, batch=combat_input["batch_label"]).T.reset_index()
    return combat_input.iloc[:, :5].merge(corrected, on="SampleID", how="inner")


def export_lc_nonlc(corrected: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Restore the LC/non-LC sample set and output schema."""
    subset = corrected.loc[corrected["SampleID"].isin(source["SampleID"])].copy()
    drop_columns = [
        column for column in ("DeviceID", "AnalysisTime", "batch_label") if column in subset
    ]
    subset = subset.drop(drop_columns, axis=1)
    subset["label"] = subset["label"].map(
        lambda value: value if value in ["LC", "ENV"] else "nonLC"
    )
    return subset
