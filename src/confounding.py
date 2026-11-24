"""Confounding-adjustment modelling for the LC/non-LC marker model.

The source notebook fitted one XGBoost model per adjustment scheme: dropping
one of the base covariates (age, gender) or replacing them with a single
clinical covariate's dummy encoding, plus one model with every clinical
dummy. This module reproduces that loop with an explicit, data-driven
adjustment specification instead of repeated notebook cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import pandas as pd

from .evaluation import (
    bootstrap_auc_interval,
    evaluate_predictions,
    find_sensitivity_cutoff,
    require_columns,
)

BASE_COVARIATES = ("age", "gender")
IDENTIFIER_COLUMNS = ("SampleID", "label")


@dataclass(frozen=True)
class AdjustmentSpec:
    """One confounding-adjustment scheme.

    ``drop_columns`` are base covariates removed from the RFECV modelling
    frame. ``dummy_prefix`` selects covariate dummy columns from the exported
    confounder matrices by name prefix; ``use_all_dummies`` merges every
    clinical dummy except the base covariates and project cohort dummies.
    """

    name: str
    drop_columns: tuple[str, ...] = ()
    dummy_prefix: str | None = None
    use_all_dummies: bool = False


DEFAULT_ADJUSTMENTS: tuple[AdjustmentSpec, ...] = (
    AdjustmentSpec(name="age", drop_columns=("gender",)),
    AdjustmentSpec(name="gender", drop_columns=("age",)),
    AdjustmentSpec(name="BMI", drop_columns=BASE_COVARIATES, dummy_prefix="BMI_"),
    AdjustmentSpec(name="dining", drop_columns=BASE_COVARIATES, dummy_prefix="dining_status_"),
    AdjustmentSpec(name="smoking", drop_columns=BASE_COVARIATES, dummy_prefix="smoke_hist_"),
    AdjustmentSpec(
        name="secondhand_smoking",
        drop_columns=BASE_COVARIATES,
        dummy_prefix="second-hand_smoke_status_",
    ),
    AdjustmentSpec(name="cooking", drop_columns=BASE_COVARIATES, dummy_prefix="env_smoke_status_"),
    AdjustmentSpec(
        name="hazardous_substance",
        drop_columns=BASE_COVARIATES,
        dummy_prefix="hazardous_substance_exposure_",
    ),
    AdjustmentSpec(name="family_Ca", drop_columns=BASE_COVARIATES, dummy_prefix="family_Ca_hist_"),
    AdjustmentSpec(
        name="lung_disease", drop_columns=BASE_COVARIATES, dummy_prefix="lung_disease_hist_"
    ),
    AdjustmentSpec(
        name="other_disease", drop_columns=BASE_COVARIATES, dummy_prefix="other_disease_hist_"
    ),
    AdjustmentSpec(name="self_Ca", drop_columns=BASE_COVARIATES, dummy_prefix="Ca_hist_"),
    AdjustmentSpec(name="ALL", use_all_dummies=True),
)

DEFAULT_PROJECT_PREFIXES = ("proj_",)


def split_discovery_partitions(
    training_set: pd.DataFrame,
    train_predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the discovery modelling frame using the evaluation-fold membership."""
    require_columns(training_set, ["SampleID", "label"])
    require_columns(train_predictions, ["SampleID"])
    require_columns(validation_predictions, ["SampleID"])
    train_frame = training_set.loc[
        training_set["SampleID"].isin(train_predictions["SampleID"])
    ].copy()
    valid_frame = training_set.loc[
        training_set["SampleID"].isin(validation_predictions["SampleID"])
    ].copy()
    if train_frame.empty or valid_frame.empty:
        raise ValueError(
            "The train/validation prediction files do not partition the discovery set."
        )
    overlap = set(train_frame["SampleID"]) & set(valid_frame["SampleID"])
    if overlap:
        raise ValueError(
            f"{len(overlap)} samples appear in both the train and validation partitions."
        )
    return train_frame, valid_frame


def select_dummy_columns(
    confounder_matrix: pd.DataFrame,
    spec: AdjustmentSpec,
    project_prefixes: Sequence[str] = DEFAULT_PROJECT_PREFIXES,
) -> list[str]:
    """Select the confounder-matrix columns attached for an adjustment."""
    columns = [str(column) for column in confounder_matrix.columns if column != "SampleID"]
    if spec.use_all_dummies:
        excluded = {"age", "gender_F", "gender_M"}
        return [
            column
            for column in columns
            if column not in excluded
            and not any(column.startswith(prefix) for prefix in project_prefixes)
        ]
    if spec.dummy_prefix is None:
        return []
    selected = [column for column in columns if column.startswith(spec.dummy_prefix)]
    if not selected:
        raise ValueError(
            f"No confounder-matrix columns match prefix '{spec.dummy_prefix}'."
        )
    return selected


def attach_dummy_covariates(
    frame: pd.DataFrame,
    confounder_matrix: pd.DataFrame,
    spec: AdjustmentSpec,
    project_prefixes: Sequence[str] = DEFAULT_PROJECT_PREFIXES,
) -> pd.DataFrame:
    """Merge the selected dummy columns and rename them to ``CFD_i``."""
    dummy_columns = select_dummy_columns(confounder_matrix, spec, project_prefixes)
    if not dummy_columns:
        return frame.copy()
    attached = frame.merge(
        confounder_matrix[["SampleID", *dummy_columns]], on="SampleID", how="left"
    )
    renamed = [f"CFD_{index}" for index in range(len(dummy_columns))]
    attached.columns = attached.columns[: -len(dummy_columns)].tolist() + renamed
    if attached[renamed].isna().any().any():
        raise ValueError(f"Dummy covariates for adjustment '{spec.name}' contain missing values.")
    return attached


def run_adjustment(
    spec: AdjustmentSpec,
    model_factory: Callable[..., Any],
    parameters: dict[str, Any],
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    external_frame: pd.DataFrame,
    discovery_confounders: pd.DataFrame,
    external_confounders: pd.DataFrame,
    min_sensitivity: float = 0.95,
    cutoff_steps: int = 100,
    bootstrap_rounds: int = 1000,
    random_state: int | None = None,
    project_prefixes: Sequence[str] = DEFAULT_PROJECT_PREFIXES,
) -> dict[str, Any]:
    """Fit one adjustment model and evaluate it on all three partitions."""
    train = train_frame.drop(columns=[c for c in spec.drop_columns if c in train_frame.columns])
    valid = valid_frame.drop(columns=[c for c in spec.drop_columns if c in valid_frame.columns])
    external = external_frame.drop(
        columns=[c for c in spec.drop_columns if c in external_frame.columns]
    )
    train = attach_dummy_covariates(train, discovery_confounders, spec, project_prefixes)
    valid = attach_dummy_covariates(valid, discovery_confounders, spec, project_prefixes)
    external = attach_dummy_covariates(external, external_confounders, spec, project_prefixes)

    feature_columns = [column for column in train.columns if column not in IDENTIFIER_COLUMNS]
    if list(external.columns) != list(train.columns):
        raise ValueError(
            f"External feature columns do not match training columns for '{spec.name}'."
        )
    model = model_factory(**parameters)
    model.fit(train[feature_columns], train["label"])

    predictions: dict[str, pd.DataFrame] = {}
    for partition, frame in [("train", train), ("valid", valid), ("external", external)]:
        scores = model.predict_proba(frame[feature_columns])[:, 1]
        predictions[partition] = pd.DataFrame(
            {
                "SampleID": frame["SampleID"].tolist(),
                "label": frame["label"].tolist(),
                "pred": scores,
            }
        )

    cutoff = find_sensitivity_cutoff(
        predictions["valid"], min_sensitivity=min_sensitivity, steps=cutoff_steps
    )

    evaluations: dict[str, dict[str, Any]] = {}
    for partition in ("valid", "external"):
        frame = predictions[partition]
        result = evaluate_predictions(frame, cutoff)
        auc, auc_lower, auc_upper = bootstrap_auc_interval(
            frame["label"].to_numpy(dtype=int),
            frame["pred"].to_numpy(dtype=float),
            n_bootstrap=bootstrap_rounds,
            random_state=random_state,
        )
        result["auc_lb"] = auc_lower
        result["auc_ub"] = auc_upper
        evaluations[partition] = result

    return {
        "adjustment": spec.name,
        "cutoff": cutoff,
        "predictions": predictions,
        "evaluations": evaluations,
    }
