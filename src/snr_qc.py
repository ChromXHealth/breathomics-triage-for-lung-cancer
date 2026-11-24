"""Signal-to-noise quality control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from .evaluation import require_columns


@dataclass
class QualityControlResult:
    """Tables produced by SNR quality control."""

    callmap: pd.DataFrame
    voc_call_rates: pd.DataFrame
    sample_call_rates: pd.DataFrame
    retained_voc: pd.DataFrame
    retained_callmap: pd.DataFrame
    retained_features: list[str]


def build_callmap(snr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Convert SNR values to binary calls using a strict greater-than rule."""
    require_columns(snr, ["SampleID", "label"])
    calls = (snr.iloc[:, 2:] > threshold).astype(int)
    return pd.concat([snr.iloc[:, :2].copy(), calls], axis=1)


def calculate_voc_call_rates(callmap: pd.DataFrame) -> pd.DataFrame:
    """Calculate call rates by label and across all samples."""
    grouped = callmap.iloc[:, 1:].groupby("label")
    rates = grouped.sum().T / grouped.size()
    rates["ALL"] = callmap.iloc[:, 2:].sum() / callmap.shape[0]
    rates.index = rates.index.astype(str)
    return rates


def select_retained_features(
    call_rates: pd.DataFrame, threshold: float
) -> list[str]:
    """Retain features meeting the call-rate threshold in at least one column."""
    keep = (call_rates >= threshold).any(axis=1)
    return call_rates.index[keep].astype(str).tolist()


def calculate_sample_call_rates(callmap: pd.DataFrame) -> pd.DataFrame:
    """Calculate sample call rates across all QC feature columns."""
    feature_columns = list(callmap.columns[2:])
    if not feature_columns:
        raise ValueError("At least one QC feature is required for sample call rates.")
    rates = callmap[feature_columns].sum(axis=1) / len(feature_columns)
    return pd.DataFrame(
        {"SampleID": callmap["SampleID"].to_numpy(), "Sample CallRate": rates.to_numpy()},
        index=callmap.index,
    )


def apply_snr_qc(
    voc: pd.DataFrame,
    snr: pd.DataFrame,
    voc_threshold: float,
    sample_threshold: float,
    snr_threshold: float,
) -> QualityControlResult:
    """Apply the manuscript feature- and sample-level SNR filtering sequence."""
    required = ["SampleID", "label"]
    require_columns(voc, required)
    if len(voc) != len(snr):
        raise ValueError("VOC and SNR tables must contain the same number of rows.")

    callmap = build_callmap(snr, snr_threshold)
    call_rates = calculate_voc_call_rates(callmap)
    features = select_retained_features(call_rates, voc_threshold)
    filtered_callmap = callmap[["SampleID", "label"] + features].copy()
    sample_rates = calculate_sample_call_rates(filtered_callmap)
    keep_sample = sample_rates["Sample CallRate"] >= sample_threshold

    retained_columns = ["SampleID", "label"] + features
    return QualityControlResult(
        callmap=callmap,
        voc_call_rates=call_rates,
        sample_call_rates=sample_rates,
        retained_voc=voc.loc[keep_sample, retained_columns].copy(),
        retained_callmap=filtered_callmap.loc[keep_sample].copy(),
        retained_features=features,
    )


def filter_features(
    frame: pd.DataFrame, features: Sequence[str]
) -> pd.DataFrame:
    """Apply retained feature identifiers to another measurement table."""
    columns = ["SampleID", "label"] + list(features)
    require_columns(frame, columns)
    return frame[columns].copy()
