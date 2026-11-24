"""Workflow 03: analyze environmental VOCs for LC/non-LC analysis.

Run as ``python -m workflows.03_environmental_voc`` from this repository's
root after workflow 02. Every input and output path must be supplied by the
caller.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
from pathlib import Path

import pandas as pd

from src.environmental_voc import (
    add_voc_information,
    analyze_sites,
    classify_environmental_voc,
    environmental_intersection,
    label_breath_and_environment,
    prepare_sample_locations,
    remove_environmental_voc,
)
from src.io import write_csv
from src.runlog import final_outputs, note, step

OUTPUT_SUBDIR = "03_environmental_voc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected-voc", type=Path, required=True)
    parser.add_argument("--breath-locations", type=Path, required=True)
    parser.add_argument("--environment-locations", type=Path, required=True)
    parser.add_argument("--voc-info", type=Path, required=True)
    parser.add_argument("--sample-id-column", default="SampleID")
    parser.add_argument("--location-column", default="SampleLoc")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output"),
        help="Base output directory; results are written under {output-dir}/"
        f"{OUTPUT_SUBDIR}/.",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir / OUTPUT_SUBDIR
    return args


def run(args: argparse.Namespace) -> None:
    """Identify environmental VOCs and remove them from LC/non-LC data."""
    TOTAL_STEPS = 4

    step(1, TOTAL_STEPS, "Load VOC matrix and sample locations")
    source = pd.read_csv(args.corrected_voc)
    voc = label_breath_and_environment(source)
    note(f"Loaded {len(voc)} samples ({voc['label'].nunique()} origins).")

    location_frames = [pd.read_csv(args.breath_locations), pd.read_csv(args.environment_locations)]
    raw_locations = pd.concat(location_frames, ignore_index=True)
    locations = prepare_sample_locations(
        raw_locations,
        voc["SampleID"],
        args.sample_id_column,
        args.location_column,
    )

    step(2, TOTAL_STEPS, "Sampling-site analysis")
    metrics, subsets = analyze_sites(voc, locations)
    write_csv(metrics, args.output_dir / "ENV_BR_metrics.csv")
    for site, subset in subsets.items():
        write_csv(subset, args.output_dir / "analysis_dset" / f"{site}.csv")
    note(f"Analyzed {len(subsets)} sampling sites.")

    step(3, TOTAL_STEPS, "Environmental VOC classification")
    joint, _, _ = classify_environmental_voc(metrics)
    joint = add_voc_information(joint, pd.read_excel(args.voc_info))
    write_csv(joint, args.output_dir / "ENV_BR_joint_label.csv")
    note(f"Classified {len(joint)} VOCs.")

    step(4, TOTAL_STEPS, "Remove environmental VOCs")
    environmental = environmental_intersection(joint)
    union = joint.loc[(joint["ROCAUC"] == "ENV") | (joint["FDR"] == "ENV")]
    write_csv(union, args.output_dir / "ENV_VOC_by_union.csv")

    output = remove_environmental_voc(source, environmental)
    write_csv(output, args.output_dir / "evoc_free_LC_nonLC.csv")
    note(f"Retained {len(output)} rows and {output.shape[1]} columns for LC/non-LC analysis.")
    final_outputs(
        [
            (
                str(args.output_dir / "evoc_free_LC_nonLC.csv"),
                "LC/non-LC matrix with environmental VOCs removed; input to workflow 04",
            ),
            (
                str(args.output_dir / "ENV_VOC_by_union.csv"),
                "identified environmental VOCs (union of ROCAUC/FDR criteria)",
            ),
        ]
    )


if __name__ == "__main__":
    run(parse_args())
