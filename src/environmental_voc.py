"""Environmental volatile organic compound analysis."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, ttest_ind
from sklearn.metrics import roc_auc_score

from .evaluation import require_columns


def label_breath_and_environment(frame: pd.DataFrame) -> pd.DataFrame:
    """Map all non-environmental records to the breath reference group."""
    labelled = frame.copy()
    labelled["label"] = labelled["label"].map(
        lambda value: "ENV" if value == "ENV" else "BR"
    )
    return labelled


def prepare_sample_locations(
    frame: pd.DataFrame,
    sample_ids: pd.Series,
    sample_id_column: str = "SampleID",
    location_column: str = "SampleLoc",
) -> pd.DataFrame:
    """Standardize sample-location mappings and remove unknown locations."""
    require_columns(frame, [sample_id_column, location_column])
    locations = frame.loc[frame[sample_id_column].isin(sample_ids)].copy()
    locations = locations[[sample_id_column, location_column]].rename(
        columns={sample_id_column: "SampleID", location_column: "SampleLoc"}
    )
    locations = locations.fillna("UNK")
    return locations.loc[locations["SampleLoc"] != "UNK"].copy()


def iter_analysis_sites(
    voc: pd.DataFrame, locations: pd.DataFrame, minimum_group_size: int = 3
):
    """Yield site datasets containing both groups with sufficient observations."""
    for site in locations["SampleLoc"].unique():
        identifiers = locations.loc[locations["SampleLoc"] == site, "SampleID"]
        subset = voc.loc[voc["SampleID"].isin(identifiers)].copy()
        if len(subset["label"].unique()) != 2:
            continue
        if (subset["label"].value_counts() <= minimum_group_size).any():
            continue
        yield str(site), subset


def analyze_site(site: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate Welch t-test, FDR, and direction-independent AUC by feature."""
    results = defaultdict(list)
    for feature in frame.columns[2:]:
        labels = (frame["label"] == "ENV").astype(int)
        values = frame[feature]
        environmental = values.loc[labels == 1].to_numpy()
        breath = values.loc[labels != 1].to_numpy()
        _, pvalue = ttest_ind(environmental, breath)
        auc = roc_auc_score(labels, values)
        if auc < 0.5:
            auc = roc_auc_score(labels.map(lambda value: abs(value - 1)), values)

        results["VOC ID"].append(str(feature))
        results[f"pval_{site}"].append(pvalue)
        results[f"rocauc_{site}"].append(auc)

    result = pd.DataFrame(results)
    result[f"fdr_{site}"] = false_discovery_control(result[f"pval_{site}"])
    return result


def analyze_sites(voc: pd.DataFrame, locations: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Analyze all eligible sampling sites and merge metrics by VOC identifier."""
    merged = None
    subsets = {}
    for site, subset in iter_analysis_sites(voc, locations):
        subsets[site] = subset
        result = analyze_site(site, subset)
        merged = result if merged is None else merged.merge(result, on="VOC ID")
    if merged is None:
        raise ValueError("No sampling site meets the environmental-analysis criteria.")
    return merged, subsets


def classify_environmental_voc(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Classify VOCs by the manuscript ROC-AUC and FDR decision rules."""
    roc_columns = [column for column in metrics if column.startswith("rocauc")]
    fdr_columns = [column for column in metrics if column.startswith("fdr")]

    roc = metrics[["VOC ID"] + roc_columns].copy()
    roc[roc_columns] = (roc[roc_columns] > 0.8).astype(int)
    roc.insert(1, "category", np.where(roc[roc_columns].sum(axis=1) > 0, "BR", "ENV"))

    fdr = metrics[["VOC ID"] + fdr_columns].copy()
    fdr[fdr_columns] = (fdr[fdr_columns] < 0.05).astype(int)
    fdr.insert(1, "category", np.where(fdr[fdr_columns].sum(axis=1) > 1, "BR", "ENV"))

    joint = roc[["VOC ID", "category"]].merge(
        fdr[["VOC ID", "category"]], on="VOC ID"
    ).rename(columns={"category_x": "ROCAUC", "category_y": "FDR"})
    return joint, roc, fdr


def add_voc_information(joint: pd.DataFrame, information: pd.DataFrame) -> pd.DataFrame:
    """Append compound metadata and preserve the source float32 conversion."""
    require_columns(information, ["Alignment ID"])
    info = information.rename(columns={"Alignment ID": "VOC ID"}).copy()
    info["VOC ID"] = info["VOC ID"].astype(str)
    if info["VOC ID"].duplicated().any():
        raise ValueError("Alignment ID must be unique in the VOC annotation table.")
    output = info.merge(joint, on="VOC ID", how="right", validate="one_to_one")
    for column in ("Average Rt(min)", "Quant mass", "S/N average"):
        output[column] = output[column].astype(np.float32)
    return output


def environmental_intersection(joint: pd.DataFrame) -> pd.Series:
    """Return VOCs classified as environmental by both decision rules."""
    return joint.loc[(joint["ROCAUC"] == "ENV") & (joint["FDR"] == "ENV"), "VOC ID"]


def remove_environmental_voc(frame: pd.DataFrame, identifiers: pd.Series) -> pd.DataFrame:
    """Remove environmental VOCs and environmental sample records."""
    output = frame.drop(columns=frame.columns.intersection(identifiers.astype(str)))
    return output.loc[output["label"] != "ENV"].copy()
