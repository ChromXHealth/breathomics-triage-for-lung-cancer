"""Workflow 01: perform SNR quality control and sensitivity correction.

Run as ``python -m workflows.01_snr_qc`` from this repository's root. Every
data path must be supplied by the caller.

Upstream data preparation is assumed complete before this workflow runs:
instrument output has been converted to sample-by-feature tables, clinical
eligibility filtering has been applied, and breath and environmental
measurements have been separated into distinct datasets (see
``docs/data_schema.md``).
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.io import write_csv
from src.runlog import final_outputs, note, step
from src.sensitivity_correction import correct_sensitivity, prepare_environmental_input
from src.snr_qc import apply_snr_qc, filter_features

DEFAULT_PREPROCESSING_CONFIG = Path("configs/preprocessing.yaml")
OUTPUT_SUBDIR = "01_snr_qc"


def load_snr_qc_config(path: Path) -> dict:
    """Read the ``snr_qc`` parameter block from the preprocessing config."""
    with open(path) as handle:
        return yaml.safe_load(handle)["snr_qc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-input", type=Path, required=True)
    parser.add_argument("--snr-input", type=Path, required=True)
    parser.add_argument("--environmental-voc", type=Path, required=True)
    parser.add_argument("--followup-voc", type=Path, required=True)
    parser.add_argument("--analysis-time", type=Path, required=True)
    parser.add_argument("--sensitivity-factors", type=Path, required=True)
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=DEFAULT_PREPROCESSING_CONFIG,
        help="YAML file with the snr_qc parameter block; defaults to the repository config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output"),
        help="Base output directory; results are written under {output-dir}/"
        f"{OUTPUT_SUBDIR}/.",
    )
    parser.add_argument(
        "--exclude-sample",
        action="append",
        default=[],
        help="Sample identifier to exclude before correction; repeat as needed.",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir / OUTPUT_SUBDIR
    return args


def attach_analysis_time(frame: pd.DataFrame, analysis_time: pd.DataFrame) -> pd.DataFrame:
    """Attach acquisition metadata while retaining every measurement record."""
    return analysis_time.merge(frame, on="SampleID", how="right")


def run(args: argparse.Namespace) -> None:
    """Execute LC/non-LC quality control and sensitivity correction."""
    TOTAL_STEPS = 4

    step(1, TOTAL_STEPS, "Load inputs and QC configuration")
    analysis_time = pd.read_csv(args.analysis_time)
    factors = pd.read_csv(args.sensitivity_factors)
    environmental_raw = pd.read_csv(args.environmental_voc)
    qc_config = load_snr_qc_config(args.preprocessing_config)
    voc = pd.read_csv(args.voc_input)
    snr = pd.read_csv(args.snr_input)
    note(
        f"Loaded {len(voc)} LC/non-LC samples ({voc.shape[1]} columns), "
        f"{len(environmental_raw)} environmental samples, and {len(factors)} "
        f"sensitivity factors covering {factors['DeviceID'].nunique()} device(s)."
    )

    step(2, TOTAL_STEPS, "SNR quality control")
    qc = apply_snr_qc(
        voc,
        snr,
        voc_threshold=qc_config["voc_call_rate_threshold"],
        sample_threshold=qc_config["sample_call_rate_threshold"],
        snr_threshold=qc_config["snr_threshold"],
    )
    note(f"Retained {len(qc.retained_features)} features and {len(qc.retained_voc)} samples.")

    write_csv(qc.voc_call_rates, args.output_dir / "callrate_voc_LC_nonLC.csv")
    write_csv(qc.sample_call_rates, args.output_dir / "callrate_sample_LC_nonLC.csv")
    write_csv(qc.retained_voc, args.output_dir / "SNR_QC_voc_LC_nonLC.csv")
    write_csv(qc.retained_callmap, args.output_dir / "SNR_QC_callmap_LC_nonLC.csv")
    write_csv(qc.retained_voc, args.output_dir / "imputed_voc_LC_nonLC.csv")

    step(3, TOTAL_STEPS, "Sensitivity correction (LC/non-LC + environmental)")
    environmental = prepare_environmental_input(environmental_raw, qc.retained_voc.columns)
    environmental = attach_analysis_time(environmental, analysis_time)
    samples = attach_analysis_time(qc.retained_voc, analysis_time)
    samples = pd.concat([environmental, samples], ignore_index=True)
    samples = samples.loc[~samples["SampleID"].isin(args.exclude_sample)].copy()
    corrected = correct_sensitivity(samples, factors)
    write_csv(corrected, args.output_dir / "sens_corrected_voc_LC_nonLC_ENV.csv")
    note(f"Corrected {len(corrected)} samples (breath + environmental).")

    step(4, TOTAL_STEPS, "Follow-up correction")
    followup = pd.read_csv(args.followup_voc)
    followup.insert(1, "label", "postOp")
    followup = filter_features(followup, qc.retained_features)
    followup = attach_analysis_time(followup, analysis_time)
    corrected_followup = correct_sensitivity(followup, factors)
    write_csv(corrected_followup, args.output_dir / "sens_corrected_postOp_voc.csv")
    note(f"Corrected {len(corrected_followup)} follow-up samples.")

    final_outputs(
        [
            (
                str(args.output_dir / "sens_corrected_voc_LC_nonLC_ENV.csv"),
                "sensitivity-corrected breath + environmental matrix; input to workflow 02",
            ),
            (
                str(args.output_dir / "sens_corrected_postOp_voc.csv"),
                "sensitivity-corrected postoperative table; used to build the postOp model matrix",
            ),
            (
                str(args.output_dir / "callrate_voc_LC_nonLC.csv"),
                "QC diagnostic: feature-level call rates",
            ),
            (
                str(args.output_dir / "callrate_sample_LC_nonLC.csv"),
                "QC diagnostic: sample-level call rates",
            ),
        ]
    )


if __name__ == "__main__":
    run(parse_args())
