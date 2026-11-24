"""Workflow 04: create LC/non-LC discovery and external datasets.

Run as ``python -m workflows.04_data_split`` from this repository's root after
workflow 03. The cohort split is caller-supplied either as repeated
command-line arguments or through a ``--split-config`` YAML file (not both).
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
from pathlib import Path

import pandas as pd

from src.io import write_csv
from src.runlog import final_outputs, note, step
from src.splitting import (
    load_cohort_split,
    normalize_clinical_fields,
    prepare_analysis_records,
    split_discovery_and_external,
)

OUTPUT_SUBDIR = "04_data_split"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-input", type=Path, required=True)
    parser.add_argument("--clinical-input", type=Path, required=True)
    parser.add_argument(
        "--split-config",
        type=Path,
        default=None,
        help="Optional YAML cohort-split config with 'discovery' and 'external' "
        "blocks (keys: projects, id_prefixes). Cannot be combined with the "
        "repeated split arguments below; without it, --discovery-project and "
        "--external-project are required (id-prefix lists may be omitted).",
    )
    parser.add_argument("--discovery-project", action="append", default=None)
    parser.add_argument("--external-project", action="append", default=None)
    parser.add_argument("--discovery-id-prefix", action="append", default=None)
    parser.add_argument("--external-id-prefix", action="append", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output"),
        help="Base output directory; results are written under {output-dir}/"
        f"{OUTPUT_SUBDIR}/.",
    )
    args = parser.parse_args()

    cli_given = any(
        [
            args.discovery_project,
            args.external_project,
            args.discovery_id_prefix,
            args.external_id_prefix,
        ]
    )
    if args.split_config is not None:
        if cli_given:
            parser.error(
                "--split-config cannot be combined with --discovery/--external "
                "project or id-prefix arguments."
            )
        args.split = load_cohort_split(args.split_config)
    else:
        if not (args.discovery_project and args.external_project):
            parser.error(
                "Either --split-config or both --discovery-project and "
                "--external-project are required."
            )
        args.split = {
            "discovery": {
                "projects": args.discovery_project,
                "id_prefixes": args.discovery_id_prefix or [],
            },
            "external": {
                "projects": args.external_project,
                "id_prefixes": args.external_id_prefix or [],
            },
        }
    args.output_dir = args.output_dir / OUTPUT_SUBDIR
    return args


def run(args: argparse.Namespace) -> None:
    """Prepare records and write discovery and external datasets."""
    TOTAL_STEPS = 3

    step(1, TOTAL_STEPS, "Align VOC and clinical records")
    voc, clinical = prepare_analysis_records(
        pd.read_csv(args.voc_input), pd.read_csv(args.clinical_input)
    )
    note(f"Aligned {len(voc)} VOC samples with {len(clinical)} clinical records.")

    step(2, TOTAL_STEPS, "Normalize clinical fields")
    clinical = normalize_clinical_fields(clinical)

    step(3, TOTAL_STEPS, "Split discovery and external cohorts")
    discovery, discovery_clinical, external, external_clinical = split_discovery_and_external(
        voc,
        clinical,
        args.split["discovery"]["projects"],
        args.split["external"]["projects"],
        args.split["discovery"]["id_prefixes"],
        args.split["external"]["id_prefixes"],
    )
    note(f"Discovery: {len(discovery)} samples; external: {len(external)} samples.")
    if args.split_config is not None:
        note(f"Cohort split loaded from {args.split_config}.")

    outputs = {
        "VOC_LC_nonLC_discovery.csv": discovery,
        "cinfo_LC_nonLC_discovery.csv": discovery_clinical,
        "VOC_LC_nonLC_external.csv": external,
        "cinfo_LC_nonLC_external.csv": external_clinical,
    }
    for filename, frame in outputs.items():
        write_csv(frame, args.output_dir / filename)
    final_outputs(
        [
            (
                str(args.output_dir / "VOC_LC_nonLC_discovery.csv"),
                "discovery VOC matrix; input to workflows 05 and 06",
            ),
            (
                str(args.output_dir / "cinfo_LC_nonLC_discovery.csv"),
                "discovery clinical table; input to workflows 05 and 06",
            ),
            (
                str(args.output_dir / "VOC_LC_nonLC_external.csv"),
                "external VOC matrix; input to workflow 06",
            ),
            (
                str(args.output_dir / "cinfo_LC_nonLC_external.csv"),
                "external clinical table; input to workflows 05 and 06",
            ),
        ]
    )


if __name__ == "__main__":
    run(parse_args())
