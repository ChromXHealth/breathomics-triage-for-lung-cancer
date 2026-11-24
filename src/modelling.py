"""Model-matrix construction, feature selection, and cross-validated prediction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.svm import SVC

from .evaluation import require_columns


def _build_xgboost(**parameters: Any) -> Any:
    """Instantiate XGBoost lazily so CLI help works without it installed."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for the xgboost model. Install it via the "
            "project environment (see environment.yml)."
        ) from exc
    return XGBClassifier(**parameters)


MODEL_REGISTRY: dict[str, Callable[..., Any]] = {
    "logistic": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "knn": KNeighborsClassifier,
    "xgboost": _build_xgboost,
    "svm": SVC,
}

# Default hyperparameters taken from the source modelling notebook
# From the source analysis: the parameter-search grids collapse to a single
# candidate per family, i.e. the recorded search end-point is used directly.
# No runtime hyperparameter search is performed by any workflow.
DEFAULT_MODEL_PARAMETERS: dict[str, dict[str, Any]] = {
    # max_iter raised above the sklearn default so the saga/elasticnet fit
    # converges quietly on this dataset size.
    "logistic": {
        "C": 0.8,
        "penalty": "elasticnet",
        "solver": "saga",
        "l1_ratio": 0.45,
        "max_iter": 5000,
    },
    "random_forest": {
        "oob_score": True,
        "bootstrap": True,
        "n_estimators": 80,
        "max_depth": 14,
        "min_samples_leaf": 2,
        "min_samples_split": 3,
        "max_features": 5,
    },
    "knn": {"n_neighbors": 18, "weights": "distance", "p": 1},
    "xgboost": {
        "eta": 0.3,
        "gamma": 0,
        "n_estimators": 500,
        "learning_rate": 0.02,
        "subsample": 0.5,
        "colsample_bytree": 0.6,
        "max_depth": 14,
    },
    # probability=True is required because the evaluation pipeline reads
    # predict_proba scores.
    "svm": {
        "C": 1,
        "kernel": "rbf",
        "gamma": "scale",
        "class_weight": "balanced",
        "probability": True,
    },
}


def load_marker_identifiers(
    marker_table: pd.DataFrame,
    identifier_column: str = "VOC ID",
    flag_column: str | None = "marker_flag",
) -> list[str]:
    """Read ordered marker identifiers from an authorized marker table."""
    require_columns(marker_table, [identifier_column])
    markers = marker_table.copy()
    if flag_column and flag_column in markers.columns:
        markers = markers.loc[markers[flag_column].astype(bool)].copy()
    identifiers = markers[identifier_column].astype(str).tolist()
    if not identifiers:
        raise ValueError("The marker table does not contain any selected markers.")
    return identifiers


def discovery_covariates(discovery_clinical: pd.DataFrame) -> pd.DataFrame:
    """Create discovery-standardized age and the male-indicator covariate.

    The returned frame matches the source modelling contract
    ``SampleID, age, gender`` where ``age`` is z-scored with the discovery
    mean and standard deviation and ``gender`` is the male indicator.
    """
    require_columns(discovery_clinical, ["SampleID", "age", "gender"])
    output = discovery_clinical[["SampleID", "age", "gender"]].copy()
    output["age"] = scale(output["age"].astype(float))
    output["gender"] = output["gender"].map(lambda value: float(value == "M"))
    return output


def external_covariates(external_clinical: pd.DataFrame) -> pd.DataFrame:
    """Create externally standardized age and a male indicator."""
    require_columns(external_clinical, ["SampleID", "age", "gender"])
    output = external_clinical[["SampleID", "age", "gender"]].copy()
    output["gender"] = output["gender"].map(lambda value: float(value == "M"))
    output["age"] = scale(output["age"].astype(float))
    return output


def select_model_frame(
    measurements: pd.DataFrame,
    markers: Sequence[str],
    covariates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select marker features and optionally attach age and gender covariates."""
    require_columns(measurements, ["SampleID", "label", *markers])
    selected = measurements[["SampleID", "label", *markers]].copy()
    if covariates is not None:
        require_columns(covariates, ["SampleID"])
        selected = selected.merge(covariates, on="SampleID", how="left")
    return selected


def prepare_postoperative_frame(postoperative_measurements: pd.DataFrame) -> pd.DataFrame:
    """Apply the source log and feature-scaling transform to postoperative data."""
    metadata_columns = ["DeviceID", "AnalysisDate", "AnalysisTime"]
    output = postoperative_measurements.drop(columns=metadata_columns, errors="ignore").copy()
    require_columns(output, ["SampleID", "label"])
    feature_columns = output.columns[2:].tolist()
    transformed = np.log10(output[feature_columns].astype(float) + 1)
    output.loc[:, feature_columns] = scale(transformed)
    return output


def select_binary_labels(frame: pd.DataFrame, positive_label: str = "LC") -> pd.DataFrame:
    """Return a copy with an explicit binary integer label."""
    require_columns(frame, ["label"])
    output = frame.copy()
    output["label"] = output["label"].map(lambda value: int(value == positive_label)).astype(int)
    return output


def run_rfecv_feature_selection(
    modelling_frame: pd.DataFrame,
    markers: Sequence[str],
    cv: int = 5,
    n_jobs: int = 5,
    random_state: int | None = None,
) -> tuple[RFECV, list[str]]:
    """Select marker features using the manuscript L1 logistic RFECV settings."""
    require_columns(modelling_frame, ["SampleID", "label", *markers])
    features = modelling_frame[list(markers)].copy()
    labels = modelling_frame["label"].map(lambda value: int(value == "LC")).astype(int)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    selector = RFECV(
        estimator=LogisticRegression(C=1, penalty="l1", solver="liblinear"),
        step=1,
        cv=splitter,
        scoring="roc_auc",
        min_features_to_select=1,
        n_jobs=n_jobs,
    )
    selector.fit(features, labels)
    support = selector.get_support(indices=True)
    selected = [str(markers[index]) for index in support]
    return selector, selected


def cross_validated_predictions(
    model_factory: Callable[..., Any],
    parameters: dict[str, Any],
    training_frame: pd.DataFrame,
    cv: int = 5,
    random_state: int | None = None,
    positive_label: str = "LC",
    independent_frames: Sequence[pd.DataFrame] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    """Generate fold-wise training, validation, and independent-set predictions."""
    require_columns(training_frame, ["SampleID", "label"])
    for frame in independent_frames:
        require_columns(frame, ["SampleID", "label"])

    encoder = LabelEncoder()
    features = training_frame.iloc[:, 2:].to_numpy(dtype=float)
    labels = encoder.fit_transform(
        training_frame["label"].map(lambda value: int(value == positive_label)).astype(int)
    )
    stratification = training_frame["label"].map(lambda value: int(value == positive_label)).astype(int).tolist()

    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    train_predictions: list[pd.DataFrame] = []
    validation_predictions: list[pd.DataFrame] = []
    independent_predictions: dict[int, list[pd.DataFrame]] = {
        index: [] for index in range(len(independent_frames))
    }

    for fold_id, (train_index, validation_index) in enumerate(splitter.split(features, stratification)):
        model = model_factory(**parameters)
        model.fit(features[train_index], labels[train_index])

        fold_train = pd.DataFrame(
            {
                "SampleID": training_frame.iloc[train_index]["SampleID"].tolist(),
                "label": training_frame.iloc[train_index]["label"].tolist(),
                "KFID": fold_id,
                "pred": model.predict_proba(features[train_index])[:, 1],
            }
        )
        fold_validation = pd.DataFrame(
            {
                "SampleID": training_frame.iloc[validation_index]["SampleID"].tolist(),
                "label": training_frame.iloc[validation_index]["label"].tolist(),
                "KFID": fold_id,
                "pred": model.predict_proba(features[validation_index])[:, 1],
            }
        )
        train_predictions.append(fold_train)
        validation_predictions.append(fold_validation)

        for independent_index, independent_frame in enumerate(independent_frames):
            independent_features = independent_frame.iloc[:, 2:].to_numpy(dtype=float)
            independent_predictions[independent_index].append(
                pd.DataFrame(
                    {
                        "SampleID": independent_frame["SampleID"].tolist(),
                        "label": independent_frame["label"].tolist(),
                        "KFID": fold_id,
                        "pred": model.predict_proba(independent_features)[:, 1],
                    }
                )
            )

    return (
        pd.concat(train_predictions, ignore_index=True),
        pd.concat(validation_predictions, ignore_index=True),
        {key: pd.concat(value, ignore_index=True) for key, value in independent_predictions.items()},
    )
