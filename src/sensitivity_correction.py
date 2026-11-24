"""Instrument sensitivity correction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import require_columns

SENSITIVITY_METADATA = ["SampleID", "DeviceID", "AnalysisDate", "AnalysisTime", "label"]


def impute_low_snr_values(
    voc: pd.DataFrame, callmap: pd.DataFrame, experimental_label: str
) -> pd.DataFrame:
    """Replace low-SNR values with the group minimum among detected values.

    This is a cleaned implementation of the legacy environmental imputation
    helper. The main manuscript comparisons bypass this helper and retain the
    SNR-filtered values unchanged.
    """
    require_columns(voc, ["SampleID", "label"])
    require_columns(callmap, ["SampleID", "label"])
    if not voc.index.equals(callmap.index) or not voc.columns.equals(callmap.columns):
        raise ValueError("VOC and callmap tables must have identical indices and columns.")

    calls = callmap.iloc[:, 2:]
    valid = (voc.iloc[:, 2:] * calls).astype("float64")
    detected = valid.where(valid != 0, np.inf)
    experimental = voc["label"] == experimental_label
    experimental_min = detected.loc[experimental].min(axis=0)
    control_min = detected.loc[~experimental].min(axis=0)

    imputed = valid.copy()
    imputed.loc[experimental] = (
        valid.loc[experimental] + (1 - calls.loc[experimental]) * experimental_min
    )
    imputed.loc[~experimental] = (
        valid.loc[~experimental] + (1 - calls.loc[~experimental]) * control_min
    )
    return pd.concat([voc.iloc[:, :2].copy(), imputed], axis=1)


def prepare_environmental_input(voc: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    """Label environmental measurements and align them to analysis columns."""
    require_columns(voc, ["SampleID"])
    environmental = voc.copy()
    environmental.insert(1, "label", "ENV")
    environmental = environmental[columns]
    callmap = pd.concat(
        [environmental.iloc[:, :2], (environmental.iloc[:, 2:] > 0).astype(int)],
        axis=1,
    )
    return impute_low_snr_values(environmental, callmap, "ENV")


def correct_sensitivity(
    samples: pd.DataFrame,
    factors: pd.DataFrame,
    drop_samples_before_first_qc: bool = True,
) -> pd.DataFrame:
    """Apply device- and time-specific multiplicative sensitivity factors.

    Time intervals are open at both ends, matching the source notebook. Samples
    outside all intervals are omitted. Samples at or before the first QC time
    can optionally be retained without correction.
    """
    require_columns(samples, SENSITIVITY_METADATA)
    require_columns(factors, ["DeviceID", "AnalysisTime", "NextTime", "sensitivity_factor"])

    corrected_devices = []
    for device_id in factors["DeviceID"].unique():
        device_factors = factors.loc[factors["DeviceID"] == device_id].copy()
        device_samples = samples.loc[samples["DeviceID"] == device_id].copy()
        values = device_samples.drop(
            ["SampleID", "DeviceID", "AnalysisDate", "label"], axis=1
        )
        feature_columns = values.columns[values.columns != "AnalysisTime"]
        values = values.astype({column: "float64" for column in feature_columns})

        corrected_parts = []
        if not drop_samples_before_first_qc and not device_factors.empty:
            corrected_parts.append(
                values.loc[values["AnalysisTime"] <= device_factors["AnalysisTime"].iloc[0]]
            )

        for row in device_factors.itertuples(index=False):
            selected = values.loc[
                (values["AnalysisTime"] > row.AnalysisTime)
                & (values["AnalysisTime"] < row.NextTime)
            ].copy()
            selected.iloc[:, 1:] = selected.iloc[:, 1:] * row.sensitivity_factor
            corrected_parts.append(selected)

        if corrected_parts:
            corrected = pd.concat(corrected_parts)
            corrected_devices.append(corrected)

    if not corrected_devices:
        return samples.iloc[0:0].copy()

    corrected_values = pd.concat(corrected_devices)
    metadata = samples[SENSITIVITY_METADATA]
    return metadata.merge(
        corrected_values.iloc[:, 1:], left_index=True, right_index=True
    )
