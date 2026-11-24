"""Workflow 02: assess and correct batch effects for the LC/non-LC matrix.

Run as ``python -m workflows.02_batch_analysis`` from this repository's root
after workflow 01, feeding the sensitivity-corrected LC/non-LC + environmental
matrix directly. Every path must be supplied by the caller.

The source notebook fitted ComBat on a joint dataset built from three study
cohorts (LC/non-LC, LC/HC, LC/LN); this repository supports the LC/non-LC
cohort only, so the correction is fitted on the LC/non-LC + environmental
matrix alone (see ``docs/workflow_02.md``). ComBat correction itself is
optional: it runs only when ``--batch-config`` is supplied, otherwise the
normalized matrix is exported unchanged.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
from pathlib import Path

import pandas as pd

from src.batch_correction import (
    apply_combat,
    export_lc_nonlc,
    load_batch_rules,
    normalize_features,
    prepare_analysis_matrix,
)
from src.io import write_csv
from src.runlog import final_outputs, note, step

OUTPUT_SUBDIR = "02_batch_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="sens_corrected_voc_LC_nonLC_ENV.csv produced by workflow 01.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output"),
        help="Base output directory; results are written under {output-dir}/"
        f"{OUTPUT_SUBDIR}/.",
    )
    parser.add_argument(
        "--batch-config",
        type=Path,
        default=None,
        help="Optional YAML file with batch assignment rules. When omitted, "
        "ComBat correction is skipped and the normalized matrix is exported "
        "instead. Pass configs/batches.yaml to reproduce the source study's "
        "batch definitions.",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir / OUTPUT_SUBDIR
    return args


def run(args: argparse.Namespace) -> None:
    """Normalize the LC/non-LC matrix and apply ComBat batch correction."""
    TOTAL_STEPS = 3

    step(1, TOTAL_STEPS, "Load sensitivity-corrected matrix")
    source = pd.read_csv(args.input)
    matrix, features = prepare_analysis_matrix(source)
    note(f"Loaded {len(matrix)} samples with {len(features)} features.")

    step(2, TOTAL_STEPS, "Feature normalization")
    normalized = normalize_features(matrix)
    note("Applied log10(x + 1) and feature-wise standardization.")

    step(3, TOTAL_STEPS, "ComBat batch correction")
    if args.batch_config is not None:
        rules = load_batch_rules(args.batch_config)
        note(
            f"Batch rules from {args.batch_config} "
            f"({len(rules)} device rule set(s), including '*' fallback)."
        )
        corrected = apply_combat(normalized, rules)
        write_csv(corrected, args.output_dir / "combat_corrected_voc_LC_nonLC_ENV.csv")

        output = export_lc_nonlc(corrected, source)
        downstream_path = args.output_dir / "combat_voc_LC_nonLC_ENV.csv"
        note(f"Corrected {len(corrected)} samples; exported {len(output)} LC/non-LC rows.")
        final_outputs(
            [
                (
                    str(downstream_path),
                    "ComBat-corrected LC/non-LC + environmental matrix; input to workflow 03",
                ),
                (
                    str(args.output_dir / "combat_corrected_voc_LC_nonLC_ENV.csv"),
                    "corrected matrix with full batch metadata (reference)",
                ),
            ]
        )
    else:
        note("No --batch-config supplied; skipping ComBat correction.")
        corrected = normalized.drop(columns=["AnalysisDate"])
        write_csv(corrected, args.output_dir / "normalized_voc_LC_nonLC_ENV.csv")

        output = export_lc_nonlc(corrected, source)
        downstream_path = args.output_dir / "normalized_voc_LC_nonLC_ENV.csv"
        note(f"Exported {len(output)} LC/non-LC rows (normalized only).")
        final_outputs(
            [
                (
                    str(downstream_path),
                    "normalized LC/non-LC + environmental matrix (no ComBat); input to workflow 03",
                ),
            ]
        )
    write_csv(output, downstream_path)


if __name__ == "__main__":
    run(parse_args())
