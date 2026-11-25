"""Generate fully synthetic inputs for every public analysis workflow.

The generated identifiers, measurements, and clinical fields are random test
fixtures. They contain no participant-derived information and are not intended
for statistical or clinical interpretation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

N_DISCOVERY = 80
N_EXTERNAL = 40
N_ENVIRONMENT = 12
N_POSTOPERATIVE = 12
FEATURES = [str(index) for index in range(1, 17)]
PANEL = [str(index) for index in range(1, 13)]


def identifiers(project: str, count: int, start: int = 1) -> list[str]:
    return [f"{project}#{index:04d}" for index in range(start, start + count)]


def clinical_frame(sample_ids: list[str], labels: list[str], rng: np.random.Generator) -> pd.DataFrame:
    count = len(sample_ids)
    return pd.DataFrame(
        {
            "SampleID": sample_ids,
            "LC_nonLC": labels,
            "label_postOp": "F",
            "gender": np.where(np.arange(count) % 2, "F", "M"),
            "age": rng.integers(35, 76, count),
            "BMI": np.round(rng.uniform(18, 31, count), 1),
            "dining_status": np.where(np.arange(count) % 2, "T", "F"),
            "smoke_hist": np.where(np.arange(count) % 3 == 0, "current", "never or quit >1 yr"),
            "second-hand_smoke_status": np.where(np.arange(count) % 2, "T", "F"),
            "env_smoke_status": np.where(np.arange(count) % 3 == 0, "T", "F"),
            "hazardous_substance_exposure": np.where(np.arange(count) % 4 == 0, "T", "F"),
            "family_Ca_hist": np.where(np.arange(count) % 5 == 0, "T", "F"),
            "lung_disease_hist": np.where(np.arange(count) % 4 == 0, "T", "F"),
            "other_disease_hist": np.where(np.arange(count) % 3 == 0, "T", "F"),
            "Ca_hist": np.where(np.arange(count) % 6 == 0, "T", "F"),
            "copd": np.where(np.arange(count) % 4 == 0, "T", "F"),
        }
    )


def measurements(labels: list[str], rng: np.random.Generator) -> pd.DataFrame:
    values = rng.normal(60, 12, (len(labels), len(FEATURES)))
    for row, label in enumerate(labels):
        if label == "LC":
            values[row, :12] *= 2.8
            values[row, :12] += 35
    return pd.DataFrame(np.clip(values, 1, None), columns=FEATURES)


def imaging_frame(sample_ids: list[str], labels: list[str]) -> pd.DataFrame:
    stages = ["IA1", "IA2", "IA3", "IB", "II"]
    sizes = ["nodule_size <= 10", "10 < nodule_size <= 20", "20 < nodule_size <= 30"]
    subtypes = ["AIS", "MIA", "IA", "SCC"]
    return pd.DataFrame(
        {
            "SampleID": sample_ids,
            "ajcc_stage": [stages[index % len(stages)] if label == "LC" else "0" for index, label in enumerate(labels)],
            "nodule_size": [sizes[index % len(sizes)] for index in range(len(labels))],
            "pathology_subtype": [subtypes[index % len(subtypes)] if label == "LC" else "UNK" for index, label in enumerate(labels)],
            "nodule_type": np.where(np.arange(len(labels)) % 2, "GGN", "SN"),
            "multiplicity": np.where(np.arange(len(labels)) % 3, "SPN", "MPN"),
        }
    )


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    discovery_ids = identifiers("SYNTH-D", N_DISCOVERY)
    external_ids = identifiers("SYNTH-E", N_EXTERNAL)
    sample_ids = discovery_ids + external_ids
    labels = ["LC" if index % 2 == 0 else "nonLC" for index in range(N_DISCOVERY)]
    labels += ["LC" if index % 2 == 0 else "nonLC" for index in range(N_EXTERNAL)]
    environment_ids = identifiers("SYNTH-ENV", N_ENVIRONMENT)
    postoperative_ids = identifiers("SYNTH-POST", N_POSTOPERATIVE)

    voc = measurements(labels, rng)
    voc.insert(0, "label", labels)
    voc.insert(0, "SampleID", sample_ids)
    snr_values = rng.uniform(40, 200, (len(sample_ids), len(FEATURES)))
    snr = pd.DataFrame(snr_values, columns=FEATURES)
    snr.insert(0, "label", labels)
    snr.insert(0, "SampleID", sample_ids)

    environment = measurements(["nonLC"] * N_ENVIRONMENT, rng)
    environment.iloc[:, :12] += 1000
    environment.insert(0, "SampleID", environment_ids)
    followup = measurements(["LC"] * N_POSTOPERATIVE, rng)
    followup.insert(0, "SampleID", postoperative_ids)

    all_ids = sample_ids + environment_ids + postoperative_ids
    times = pd.date_range("2025-01-02 08:00", periods=len(all_ids), freq="2h")
    acquisition = pd.DataFrame(
        {
            "SampleID": all_ids,
            "DeviceID": "SYNTH-DEVICE",
            "AnalysisDate": times.date.astype(str),
            "AnalysisTime": times.astype(str),
        }
    )
    factors = pd.DataFrame(
        {
            "DeviceID": ["SYNTH-DEVICE"],
            "AnalysisTime": ["2025-01-01 00:00:00"],
            "NextTime": ["2025-03-01 00:00:00"],
            "sensitivity_factor": [1.0],
        }
    )

    clinical = clinical_frame(sample_ids, labels, rng)
    imaging = imaging_frame(sample_ids, labels)
    locations = pd.DataFrame(
        {
            "SampleID": sample_ids,
            "SampleLoc": np.where(np.arange(len(sample_ids)) % 2, "SITE_B", "SITE_A"),
        }
    )
    environment_locations = pd.DataFrame(
        {
            "SampleID": environment_ids,
            "SampleLoc": np.where(np.arange(N_ENVIRONMENT) % 2, "SITE_B", "SITE_A"),
        }
    )
    annotation = pd.DataFrame(
        {
            "Alignment ID": [int(feature) for feature in FEATURES],
            "Average Rt(min)": np.linspace(0.5, 12.0, len(FEATURES)),
            "Quant mass": np.linspace(40, 250, len(FEATURES)),
            "S/N average": np.linspace(30, 150, len(FEATURES)),
            "Synthetic compound": [f"Synthetic VOC {feature}" for feature in FEATURES],
        }
    )
    subtypes = pd.DataFrame(
        {
            "SampleID": sample_ids,
            "nonlc_subtype": ["benign" if label == "nonLC" else "not_applicable" for label in labels],
        }
    )
    lab_records = pd.DataFrame(
        {
            "SampleID": sample_ids,
            "dset": [value.split("#")[0] for value in sample_ids],
            "label": labels,
            "site": locations["SampleLoc"],
        }
    )
    split = {
        "discovery": {"projects": ["SYNTH-D"], "id_prefixes": []},
        "external": {"projects": ["SYNTH-E"], "id_prefixes": []},
    }
    batches = {
        "batches": {
            "*": {
                "rules": [{"batch": "Synthetic-A", "le": "2025-01-10 00:00:00"}],
                "default": "Synthetic-B",
            }
        }
    }
    markers = pd.DataFrame({"VOC ID": PANEL, "marker_flag": True})

    outputs = {
        "raw_voc_LC_nonLC.csv": voc,
        "raw_snr_LC_nonLC.csv": snr,
        "ENV-VOC-RAW.csv": environment,
        "raw_voc_followup.csv": followup,
        "acquisition_metadata.csv": acquisition,
        "sensitivity_factors.csv": factors,
        "clinical_labels.csv": clinical,
        "breath_locations.csv": locations,
        "environment_locations.csv": environment_locations,
        "imaging_labels.csv": imaging,
        "nonlc_subtypes.csv": subtypes,
        "lab_records.csv": lab_records,
        "authorized_markers.csv": markers,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    annotation.to_excel(output_dir / "voc_annotations.xlsx", index=False)
    with open(output_dir / "cohort_split.yaml", "w") as handle:
        yaml.safe_dump(split, handle, sort_keys=False)
    with open(output_dir / "batches.yaml", "w") as handle:
        yaml.safe_dump(batches, handle, sort_keys=False)
    metadata = {
        "seed": seed,
        "synthetic": True,
        "discovery_rows": N_DISCOVERY,
        "external_rows": N_EXTERNAL,
        "environment_rows": N_ENVIRONMENT,
        "postoperative_rows": N_POSTOPERATIVE,
        "features_before_environment_filter": FEATURES,
        "authorized_panel": PANEL,
    }
    (output_dir / "fixture_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Synthetic fixtures written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
