"""Dataset splitting and sample-assignment utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

from .evaluation import require_columns

SMOKING_HISTORY_MAP = {
    "never or quit >1 yr": "F",
    "current": "T",
    "quit ≤1 yr": "T",
    "T": "T",
    "F": "F",
    "UNK": "UNK",
}


def prepare_analysis_records(
    voc: pd.DataFrame, clinical: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align measurements and clinical records and apply eligibility rules."""
    require_columns(voc, ["SampleID"])
    require_columns(clinical, ["SampleID", "LC_nonLC", "label_postOp"])

    voc = voc.loc[voc["SampleID"].isin(clinical["SampleID"])].copy()
    clinical = clinical.loc[clinical["SampleID"].isin(voc["SampleID"])].copy()
    clinical = clinical.rename(columns={"LC_nonLC": "label"})
    clinical = clinical.loc[clinical["label_postOp"] == "F"].copy()
    clinical = clinical.drop(columns=["height", "weight"], errors="ignore")
    voc = voc.loc[voc["SampleID"].isin(clinical["SampleID"])].copy()
    return voc, clinical


def normalize_clinical_fields(clinical: pd.DataFrame) -> pd.DataFrame:
    """Normalize smoking history and BMI categories without imputation."""
    output = clinical.copy()
    require_columns(output, ["smoke_hist", "BMI"])
    output["smoke_hist"] = output["smoke_hist"].map(SMOKING_HISTORY_MAP)
    output["BMI"] = output["BMI"].map(map_bmi_category)
    return output


def map_bmi_category(value) -> str:
    """Map a numeric BMI value to the manuscript category."""
    if value == "UNK":
        return "UNK"
    numeric = float(value)
    if numeric < 18.5:
        return "BMI < 18.5"
    if numeric < 24:
        return "18.5 ≤ BMI < 24"
    if numeric < 28:
        return "24 ≤ BMI < 28"
    return "28 ≤ BMI"


def select_projects(frame: pd.DataFrame, projects: Sequence[str]) -> pd.DataFrame:
    """Select records by the identifier component before the first hash."""
    if not projects:
        raise ValueError("At least one project is required.")
    selected = set(projects)
    return frame.loc[
        frame["SampleID"].astype(str).map(lambda value: value.split("#")[0] in selected)
    ].copy()


def select_identifier_prefixes(frame: pd.DataFrame, prefixes: Sequence[str]) -> pd.DataFrame:
    """Select records whose identifiers start with any configured prefix."""
    if not prefixes:
        raise ValueError("At least one identifier prefix is required.")
    return frame.loc[
        frame["SampleID"].astype(str).map(
            lambda value: any(value.startswith(prefix) for prefix in prefixes)
        )
    ].copy()


def select_by_side(
    frame: pd.DataFrame, projects: Sequence[str], id_prefixes: Sequence[str]
) -> pd.DataFrame:
    """Select records for one cohort side from projects and/or ID prefixes."""
    parts = []
    if projects:
        parts.append(select_projects(frame, projects))
    if id_prefixes:
        parts.append(select_identifier_prefixes(frame, id_prefixes))
    if not parts:
        raise ValueError("A cohort side needs at least one project or identifier prefix.")
    return pd.concat(parts, ignore_index=True)


def split_discovery_and_external(
    voc: pd.DataFrame,
    clinical: pd.DataFrame,
    discovery_projects: Sequence[str],
    external_projects: Sequence[str],
    discovery_id_prefixes: Sequence[str],
    external_id_prefixes: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create discovery and external datasets with aligned clinical records."""
    discovery = select_by_side(voc, discovery_projects, discovery_id_prefixes)
    external = select_by_side(voc, external_projects, external_id_prefixes)
    discovery_clinical = clinical.loc[clinical["SampleID"].isin(discovery["SampleID"])].copy()
    external_clinical = clinical.loc[clinical["SampleID"].isin(external["SampleID"])].copy()
    return discovery, discovery_clinical, external, external_clinical


def load_cohort_split(path: Path) -> dict[str, dict[str, list[str]]]:
    """Read the discovery/external cohort split from a config file.

    Expected schema::

        discovery:
          projects: [PROJECT_A, ...]
          id_prefixes: ["PROJECT_B#0", ...]
        external:
          projects: [PROJECT_C, ...]
          id_prefixes: ["PROJECT_B#1", ...]

    ``projects`` and ``id_prefixes`` are optional per side, but each side
    must define at least one non-empty list.
    """
    with open(path) as handle:
        spec = yaml.safe_load(handle)
    if not isinstance(spec, dict):
        raise ValueError(f"Cohort split config {path} must define 'discovery' and 'external' blocks.")
    normalized: dict[str, dict[str, list[str]]] = {}
    for side in ("discovery", "external"):
        block = spec.get(side)
        if not isinstance(block, dict):
            raise ValueError(f"Cohort split config {path} must define a '{side}' block.")
        unknown = set(block) - {"projects", "id_prefixes"}
        if unknown:
            raise ValueError(f"Unknown key(s) {sorted(unknown)} in '{side}' block of {path}.")
        lists = {
            key: [str(value) for value in block.get(key) or []]
            for key in ("projects", "id_prefixes")
        }
        if not any(lists.values()):
            raise ValueError(
                f"'{side}' block of {path} needs at least one project or identifier prefix."
            )
        normalized[side] = lists
    return normalized
