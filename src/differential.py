"""Univariate differential-marker analysis for LC/non-LC discovery data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from tqdm import tqdm

from .evaluation import require_columns
from .statistics import add_fdr, build_contingency_table, direction_independent_auc


def _load_statsmodels() -> tuple[Any, Any]:
    """Import statsmodels lazily so CLI help works without it installed."""
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for GLM association analysis. Install it via "
            "the project environment (see environment.yml)."
        ) from exc
    return sm, smf

SOURCE_LABEL_COLUMNS = ("label", "label_LC", "label_LN", "label_HC", "label_postOp")
DEFAULT_CLINICAL_COVARIATES = (
    "gender",
    "age",
    "BMI",
    "dining_status",
    "smoke_hist",
    "second-hand_smoke_status",
    "env_smoke_status",
    "hazardous_substance_exposure",
    "family_Ca_hist",
    "lung_disease_hist",
    "other_disease_hist",
    "Ca_hist",
)


@dataclass(frozen=True)
class AgeStandardization:
    """Discovery-set age mean and standard deviation."""

    mean: float
    standard_deviation: float


def fit_age_standardization(
    clinical: pd.DataFrame,
    age_column: str = "age",
    ddof: int = 1,
) -> AgeStandardization:
    """Estimate age scaling from the discovery clinical table."""
    require_columns(clinical, [age_column])
    return AgeStandardization(
        mean=float(clinical[age_column].mean()),
        standard_deviation=float(clinical[age_column].std(ddof=ddof)),
    )


def standardize_age(
    clinical: pd.DataFrame,
    scaling: AgeStandardization,
    age_column: str = "age",
) -> pd.DataFrame:
    """Apply discovery-derived age scaling without modifying input fields."""
    require_columns(clinical, [age_column])
    output = clinical.copy()
    output[age_column] = (output[age_column] - scaling.mean) / scaling.standard_deviation
    return output


def encode_covariates(
    clinical: pd.DataFrame,
    scaling: AgeStandardization,
    age_column: str = "age",
    fill_missing: bool = True,
    age_first: bool = True,
) -> pd.DataFrame:
    """One-hot encode cohort and clinical fields using the source column orders.

    ``age_first=True`` reproduces the exported confounder matrix
    ``SampleID, age, encoded categorical fields``. ``age_first=False`` preserves
    the OPLS input order produced by one-hot encoding all non-identifier
    source fields before column-wise float conversion.
    """
    standardized = standardize_age(clinical, scaling, age_column)
    label_columns = [column for column in SOURCE_LABEL_COLUMNS if column in standardized.columns]

    if age_first:
        categorical_columns = [
            column
            for column in standardized.columns
            if column not in {"SampleID", age_column, *label_columns}
        ]
        categorical = standardized[categorical_columns].copy()
        if fill_missing:
            categorical = categorical.fillna("UNK")
        encoded = pd.get_dummies(categorical).astype(np.float64)
        return pd.concat(
            [
                standardized[["SampleID"]].reset_index(drop=True),
                standardized[[age_column]].reset_index(drop=True),
                encoded.reset_index(drop=True),
            ],
            axis=1,
        )

    encoded_source = standardized.drop(columns=["SampleID", *label_columns])
    encoded = pd.get_dummies(encoded_source).astype(np.float64)
    return pd.concat(
        [standardized[["SampleID"]].reset_index(drop=True), encoded.reset_index(drop=True)],
        axis=1,
    )


def prepare_analysis_frame(
    voc: pd.DataFrame,
    clinical: pd.DataFrame,
    scaling: AgeStandardization,
    covariates: Sequence[str] = DEFAULT_CLINICAL_COVARIATES,
) -> pd.DataFrame:
    """Merge measurements with discovery-standardized clinical covariates."""
    require_columns(voc, ["SampleID", "label"])
    require_columns(clinical, ["SampleID", *covariates])

    selected_clinical = standardize_age(clinical, scaling)[["SampleID", *covariates]].copy()
    output = voc.merge(selected_clinical, on="SampleID", how="inner")
    output["label"] = output["label"].map(lambda value: int(value == "LC")).astype(int)
    return output


def _safe_model_columns(covariates: Sequence[str]) -> list[str]:
    """Map covariates to formula-safe column names while preserving age."""
    return [covariate if covariate == "age" else f"CFD{index}" for index, covariate in enumerate(covariates)]


def analyze_voc_associations(
    analysis_frame: pd.DataFrame,
    precombat_frame: pd.DataFrame,
    covariates: Sequence[str] = DEFAULT_CLINICAL_COVARIATES,
) -> pd.DataFrame:
    """Compute GLM p values, adjusted AUCs, and pre/post-correction relations."""
    require_columns(analysis_frame, ["SampleID", "label", *covariates])
    require_columns(precombat_frame, ["SampleID", "label"])
    sm, smf = _load_statsmodels()

    feature_columns = [
        column
        for column in analysis_frame.columns
        if column not in {"SampleID", "label", *covariates}
    ]
    model_columns = _safe_model_columns(covariates)
    precombat = precombat_frame.loc[
        precombat_frame["SampleID"].isin(analysis_frame["SampleID"])
    ].copy()

    records: list[dict] = []
    for feature in tqdm(feature_columns, desc="VOC associations"):
        model_frame = analysis_frame[["label", feature, *covariates]].copy()
        model_frame.columns = ["label", "TARGET_VOC", *model_columns]
        covariate_terms = [
            column if column == "age" else f"C({column})" for column in model_columns
        ]
        formula = "label ~ TARGET_VOC + " + " + ".join(covariate_terms)
        model = smf.glm(formula, model_frame, family=sm.families.Binomial()).fit()

        positive_values = model_frame.loc[model_frame["label"] == 1, "TARGET_VOC"]
        negative_values = model_frame.loc[model_frame["label"] != 1, "TARGET_VOC"]
        _, relation_p_value = ttest_ind(positive_values, negative_values, alternative="greater")

        precombat_feature = precombat[["label", feature]]
        pre_positive = precombat_feature.loc[precombat_feature["label"] == "LC", feature]
        pre_negative = precombat_feature.loc[precombat_feature["label"] != "LC", feature]
        _, pre_relation_p_value = ttest_ind(pre_positive, pre_negative, alternative="greater")

        records.append(
            {
                "VOC ID": feature,
                "pval": float(model.pvalues["TARGET_VOC"]),
                "auroc": direction_independent_auc(model_frame["label"], model_frame["TARGET_VOC"]),
                "relation": "exp>ctr" if relation_p_value < 0.05 else "exp<=ctr",
                "precombat_relation": "exp>ctr" if pre_relation_p_value < 0.05 else "exp<=ctr",
            }
        )

    output = pd.DataFrame(records)
    output["fdr"] = add_fdr(output["pval"])
    return output


def analyze_covariate_associations(
    analysis_frame: pd.DataFrame,
    covariates: Sequence[str] = DEFAULT_CLINICAL_COVARIATES,
) -> pd.DataFrame:
    """Compute LC/non-LC associations for each clinical covariate."""
    require_columns(analysis_frame, ["label", *covariates])

    records: list[dict] = []
    for covariate in covariates:
        if covariate == "age":
            positive = analysis_frame.loc[analysis_frame["label"] == 1, covariate]
            negative = analysis_frame.loc[analysis_frame["label"] != 1, covariate]
            _, p_value = ttest_ind(positive, negative)
            auroc = direction_independent_auc(analysis_frame["label"], analysis_frame[covariate])
        else:
            table = build_contingency_table(analysis_frame["label"], analysis_frame[covariate])
            result = chi2_contingency(table)
            p_value = float(result.pvalue)
            auroc = None

        records.append(
            {
                "VOC ID": covariate,
                "auroc": auroc,
                "pval": p_value,
                "relation": None,
            }
        )

    output = pd.DataFrame(records)
    output["fdr"] = add_fdr(output["pval"])
    return output


def annotate_markers(
    metrics: pd.DataFrame,
    voc_info: pd.DataFrame,
    fdr_threshold: float = 0.05,
    vip_threshold: float = 1.0,
    auroc_threshold: float = 0.6,
) -> pd.DataFrame:
    """Join VOC annotations and flag markers using the manuscript thresholds."""
    require_columns(metrics, ["VOC ID", "fdr", "vip", "auroc"])

    annotations = voc_info.rename(columns={"Alignment ID": "VOC ID"}).copy()
    if "VOC ID" not in annotations.columns:
        raise ValueError("The VOC annotation table must contain 'Alignment ID' or 'VOC ID'.")
    annotations["VOC ID"] = annotations["VOC ID"].astype(str)

    output = annotations.merge(metrics, on="VOC ID", how="inner")
    output["marker_flag"] = (
        (output["fdr"] < fdr_threshold)
        & (output["vip"] > vip_threshold)
        & (output["auroc"] > auroc_threshold)
    )
    return output
