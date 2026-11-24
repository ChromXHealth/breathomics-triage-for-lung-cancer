"""Subgroup evaluation for the LC/non-LC model predictions.

The source notebook repeated the same subgroup recipe once per clinical or
imaging dimension. All recipes reduce to four immutable-field strategies:

- ``lc_stage``: label == 1 samples are grouped by an imaging/pathology field
  and each group is evaluated together with every label == 0 sample.
- ``covariate``: both classes are restricted to one value of a clinical field.
- ``nonlc_subtype``: label == 0 samples are grouped by a source label and each
  group is evaluated together with every label == 1 sample.
- ``center``: both classes are restricted to one cohort/project value.

Subgroup membership is always derived from source fields; predictions, labels,
and cohort assignments are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .evaluation import evaluate_predictions, require_columns, wilson_score_interval

NONLC_LABEL = "nonLC"
UNKNOWN_VALUE = "UNK"


@dataclass(frozen=True)
class SubgroupGroup:
    """One evaluated group within a subgroup dimension."""

    label: str
    value: Any


@dataclass(frozen=True)
class SubgroupSpec:
    """One subgroup dimension evaluated on every supplied prediction set."""

    name: str
    kind: str  # one of: lc_stage, covariate, nonlc_subtype, center
    source_column: str
    groups: tuple[SubgroupGroup, ...]
    value_mapping: Optional[dict[str, str]] = None


# Imaging/pathology dimensions (imaging label table contract).
AJCC_STAGE_GROUPS = tuple(
    SubgroupGroup(label=f"AJCC {value}", value=value)
    for value in ("0", "IA1", "IA2", "IA3", "IB", "II", "III", "IV")
)
NODULE_SIZE_MAPPING = {
    "nodule_size <= 10": "<=10mm",
    "10 < nodule_size <= 20": "10-20mm",
    "20 < nodule_size <= 30": "20-30mm",
    "30 < nodule_size": ">30mm",
}
NODULE_SIZE_GROUPS = tuple(
    SubgroupGroup(label=label, value=source)
    for source, label in NODULE_SIZE_MAPPING.items()
)
PATHOLOGY_SUBTYPE_GROUPS = tuple(
    SubgroupGroup(label=value, value=value) for value in ("AIS", "MIA", "IA", "SCC", "SCLC")
)
NODULE_TYPE_GROUPS = tuple(
    SubgroupGroup(label=label, value=source)
    for source, label in {"GGN": "GGN", "PSN": "PSN", "SN": "SN"}.items()
)
MULTIPLICITY_GROUPS = tuple(
    SubgroupGroup(label=label, value=source)
    for source, label in {"SPN": "SPN", "MPN": "MPN"}.items()
)

# Clinical covariate dimensions (clinical info table contract).
SEX_GROUPS = (
    SubgroupGroup(label="Male", value="M"),
    SubgroupGroup(label="Female", value="F"),
)
AGE_GROUPS = (
    SubgroupGroup(label="age <= 60 yrs", value="age <= 60 yrs"),
    SubgroupGroup(label="age > 60 yrs", value="age > 60 yrs"),
)
SMOKING_GROUPS = (
    SubgroupGroup(label="Smoking", value="T"),
    SubgroupGroup(label="Non-smoking", value="F"),
)
COPD_GROUPS = (
    SubgroupGroup(label="COPD", value=True),
    SubgroupGroup(label="No COPD", value=False),
)

NONLC_SUBTYPE_GROUPS = tuple(
    SubgroupGroup(label=value, value=value)
    for value in ("Healthy", "Lung nodule", "Lung disease", "Other")
)

DEFAULT_SUBGROUP_SPECS: tuple[SubgroupSpec, ...] = (
    SubgroupSpec("ajcc_stage", "lc_stage", "ajcc_stage", AJCC_STAGE_GROUPS),
    SubgroupSpec(
        "nodule_size",
        "lc_stage",
        "nodule_size",
        NODULE_SIZE_GROUPS,
        value_mapping=NODULE_SIZE_MAPPING,
    ),
    SubgroupSpec("pathology_subtype", "lc_stage", "pathology_subtype", PATHOLOGY_SUBTYPE_GROUPS),
    SubgroupSpec("nodule_type", "lc_stage", "nodule_type", NODULE_TYPE_GROUPS),
    SubgroupSpec("multiplicity", "lc_stage", "multiplicity", MULTIPLICITY_GROUPS),
    SubgroupSpec("sex", "covariate", "gender", SEX_GROUPS),
    SubgroupSpec("age", "covariate", "age_group", AGE_GROUPS),
    SubgroupSpec("smoking", "covariate", "smoke_hist", SMOKING_GROUPS),
    SubgroupSpec("copd", "covariate", "copd", COPD_GROUPS),
    SubgroupSpec("nonlc_subtype", "nonlc_subtype", "nonlc_subtype", NONLC_SUBTYPE_GROUPS),
    SubgroupSpec("center", "center", "proj", ()),  # groups derived from the data
)


def map_age_group(age: Any) -> str:
    """Map a numeric age to the manuscript age bands."""
    return "age <= 60 yrs" if float(age) < 60 else "age > 60 yrs"


def attach_imaging_labels(
    predictions: pd.DataFrame,
    imaging_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the imaging/pathology label table onto a prediction frame.

    The imaging table contract is documented in ``docs/data_schema.md``;
    cohort-specific source column names are renamed before calling.
    """
    require_columns(predictions, ["SampleID", "label", "pred"])
    available = [column for column in ("ajcc_stage", "nodule_size", "pathology_subtype", "nodule_type", "multiplicity") if column in imaging_labels.columns]
    require_columns(imaging_labels, ["SampleID", *available])
    merged = predictions.merge(imaging_labels[["SampleID", *available]], on="SampleID", how="left")
    return merged.fillna({column: UNKNOWN_VALUE for column in available})


def attach_clinical_covariates(
    predictions: pd.DataFrame,
    clinical_info: pd.DataFrame,
) -> pd.DataFrame:
    """Merge clinical covariates needed by the covariate subgroup specs."""
    require_columns(predictions, ["SampleID", "label", "pred"])
    wanted = [column for column in ("gender", "age", "smoke_hist", "copd", "proj") if column in clinical_info.columns]
    require_columns(clinical_info, ["SampleID", *wanted])
    merged = predictions.merge(clinical_info[["SampleID", *wanted]], on="SampleID", how="left")
    if "age" in merged.columns:
        merged["age_group"] = merged["age"].map(map_age_group)
    return merged


def attach_nonlc_subtypes(
    predictions: pd.DataFrame,
    subtype_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the joint non-LC subtype label table onto a prediction frame."""
    require_columns(predictions, ["SampleID", "label", "pred"])
    require_columns(subtype_labels, ["SampleID", "nonlc_subtype"])
    return predictions.merge(subtype_labels[["SampleID", "nonlc_subtype"]], on="SampleID", how="left")


def build_vis_labels(
    frame: pd.DataFrame,
    spec: SubgroupSpec,
) -> pd.Series:
    """Build the source-analysis vis_label series for one subgroup spec."""
    if spec.kind == "lc_stage":
        mapping = spec.value_mapping or {}
        label_by_value = {group.value: group.label for group in spec.groups}

        def to_group_label(value: Any) -> str:
            mapped = mapping.get(value, value)
            return str(label_by_value.get(mapped, mapped))

        values = frame[spec.source_column].map(to_group_label).fillna(UNKNOWN_VALUE)
        return pd.Series(
            np.where(frame["label"] == 1, values.astype(str), NONLC_LABEL), index=frame.index
        )
    if spec.kind == "nonlc_subtype":
        values = frame[spec.source_column].fillna(UNKNOWN_VALUE).astype(str)
        return pd.Series(
            np.where(frame["label"] == 0, values, "LC"), index=frame.index
        )
    raise ValueError(f"Spec '{spec.name}' does not use vis labels.")


def resolve_groups(spec: SubgroupSpec, frame: pd.DataFrame) -> tuple[SubgroupGroup, ...]:
    """Return the evaluated groups, deriving center groups from the data."""
    if spec.kind != "center":
        return spec.groups
    require_columns(frame, [spec.source_column])
    values = sorted(value for value in frame[spec.source_column].dropna().unique())
    return tuple(SubgroupGroup(label=str(value), value=value) for value in values)


def select_subgroup_frame(
    frame: pd.DataFrame,
    spec: SubgroupSpec,
    group: SubgroupGroup,
) -> pd.DataFrame:
    """Select the evaluated subset for one group without mutating the frame."""
    if spec.kind == "lc_stage":
        vis_label = build_vis_labels(frame, spec)
        mask = (vis_label == group.label) | (frame["label"] == 0)
        return frame.loc[mask].copy()
    if spec.kind == "covariate":
        return frame.loc[frame[spec.source_column] == group.value].copy()
    if spec.kind == "nonlc_subtype":
        vis_label = build_vis_labels(frame, spec)
        mask = (vis_label == group.label) | (frame["label"] == 1)
        return frame.loc[mask].copy()
    if spec.kind == "center":
        return frame.loc[frame[spec.source_column] == group.value].copy()
    raise ValueError(f"Unknown subgroup kind: {spec.kind}")


def evaluate_subgroup(
    frame: pd.DataFrame,
    spec: SubgroupSpec,
    group: SubgroupGroup,
    cutoff: float,
    dataset: str,
) -> Optional[dict[str, Any]]:
    """Evaluate one subgroup group at a fixed cutoff, returning a metric row."""
    subset = select_subgroup_frame(frame, spec, group)
    labels = subset["label"].to_numpy(dtype=int)
    if len(subset) == 0 or np.unique(labels).size < 2:
        return None
    result = evaluate_predictions(subset[["SampleID", "label", "pred"]], cutoff)
    row: dict[str, Any] = {
        "subgroup": spec.name,
        "group": group.label,
        "dataset": dataset,
        "cutoff": cutoff,
        "n": int(len(subset)),
        "n_lc": int((labels == 1).sum()),
        "n_nonlc": int((labels == 0).sum()),
    }
    for metric in ("AUROC", "Sensitivity", "Specificity", "PPV", "NPV"):
        row[metric] = result[metric]
        lower, upper = result["intervals"][metric]
        row[f"{metric}_lower"] = lower
        row[f"{metric}_upper"] = upper
    return row


def compute_net_benefit(
    prediction_frame: pd.DataFrame,
    steps: int = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute standardized net benefit across thresholds for a DCA curve."""
    require_columns(prediction_frame, ["label", "pred"])
    labels = prediction_frame["label"].to_numpy(dtype=int)
    scores = prediction_frame["pred"].to_numpy(dtype=float)
    prevalence = float(labels.mean())
    thresholds = np.linspace(0, 1, steps)

    net_benefits = np.empty(steps)
    treat_all = np.empty(steps)
    # The threshold=1 endpoint divides by (1 - threshold); suppress the
    # expected divide-by-zero so the DCA curve simply ends at that point.
    with np.errstate(divide="ignore", invalid="ignore"):
        for index, threshold in enumerate(thresholds):
            predicted = (scores >= threshold).astype(int)
            tp = int(((labels == 1) & (predicted == 1)).sum())
            fp = int(((labels == 0) & (predicted == 1)).sum())
            net_benefits[index] = (tp / len(labels)) - (fp / len(labels)) * (threshold / (1 - threshold))
            treat_all[index] = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
    if prevalence > 0:
        net_benefits = net_benefits / prevalence
        treat_all = treat_all / prevalence
    return net_benefits, treat_all, thresholds
