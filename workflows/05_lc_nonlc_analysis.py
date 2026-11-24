"""Workflow 05: perform LC/non-LC differential marker analysis and feature
selection.

Run as ``python -m workflows.05_lc_nonlc_analysis`` from this repository's
root after workflow 04. The screening pipeline has three stages:

1. Differential ranking applies the manuscript thresholds
   (FDR / VIP / AUROC) to flag candidate markers.
2. Qualitative identification review is a MANUAL step performed outside
   these workflows: candidate annotations are confirmed against spectral
   libraries and metabolic/physiological logic, and implausible candidates
   are excluded. See the manuscript Methods. The reviewed outcome is
   supplied here as ``--authorized-marker-table``.
3. RFECV feature selection retains the final panel from the authorized
   candidates; workflow 06 trains and evaluates the model on the selected
   tables.

Without ``--authorized-marker-table`` the flagged candidates are used
unreviewed (with an explicit log note); production runs must pass the
reviewed table. All study-data paths must be supplied by the caller.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
from pathlib import Path

import pandas as pd

from src.differential import (
    DEFAULT_CLINICAL_COVARIATES,
    analyze_covariate_associations,
    analyze_voc_associations,
    annotate_markers,
    encode_covariates,
    fit_age_standardization,
    prepare_analysis_frame,
)
from src.io import write_csv
from src.modelling import (
    discovery_covariates,
    external_covariates,
    load_marker_identifiers,
    prepare_postoperative_frame,
    run_rfecv_feature_selection,
    select_model_frame,
)
from src.opls import find_best_components, run_plsda
from src.runlog import final_outputs, note, step

OUTPUT_SUBDIR = "05_lc_nonlc_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-input", type=Path, required=True)
    parser.add_argument("--clinical-input", type=Path, required=True)
    parser.add_argument("--external-voc", type=Path, required=True)
    parser.add_argument("--external-clinical-input", type=Path, required=True)
    parser.add_argument("--postoperative-voc", type=Path, required=True)
    parser.add_argument("--precombat-voc", type=Path, required=True)
    parser.add_argument("--voc-info", type=Path, required=True)
    parser.add_argument(
        "--authorized-marker-table",
        type=Path,
        default=None,
        help="Optional human-reviewed marker table (CSV or xlsx) with a "
        "'VOC ID' column ('marker_flag' column honored if present). Records "
        "the outcome of the manual qualitative identification review that "
        "sits between threshold screening and RFECV; without it, the flagged "
        "candidates are used unreviewed and an explicit note is logged.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output"),
        help="Base output directory; results are written under {output-dir}/"
        f"{OUTPUT_SUBDIR}/.",
    )
    parser.add_argument("--positive-label", default="LC")
    parser.add_argument("--opls-search-range", type=int, default=50)
    parser.add_argument("--opls-cv", type=int, default=5)
    parser.add_argument("--cv", type=int, default=5, help="RFECV folds.")
    parser.add_argument("--rfe-jobs", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--vip-threshold", type=float, default=1.0)
    parser.add_argument("--auroc-threshold", type=float, default=0.6)
    args = parser.parse_args()
    args.output_dir = args.output_dir / OUTPUT_SUBDIR
    return args


def run(args: argparse.Namespace) -> None:
    """Execute the LC/non-LC differential, marker, and feature-selection workflow."""
    TOTAL_STEPS = 5
    voc = pd.read_csv(args.voc_input)
    clinical = pd.read_csv(args.clinical_input)
    external_voc = pd.read_csv(args.external_voc)
    external_clinical = pd.read_csv(args.external_clinical_input)
    postoperative_voc = pd.read_csv(args.postoperative_voc)
    precombat = pd.read_csv(args.precombat_voc)

    step(1, TOTAL_STEPS, "Covariate encoding")
    age_scaling = fit_age_standardization(clinical)
    discovery_encoded = encode_covariates(
        clinical, age_scaling, fill_missing=True, age_first=True
    )
    external_encoded = encode_covariates(
        external_clinical, age_scaling, fill_missing=True, age_first=True
    )
    write_csv(discovery_encoded, args.output_dir / "cfd_dummy_mat_discovery.csv")
    write_csv(external_encoded, args.output_dir / "cfd_dummy_mat_external.csv")

    step(2, TOTAL_STEPS, "Differential analysis (GLM + covariates)")
    analysis_frame = prepare_analysis_frame(
        voc, clinical, age_scaling, DEFAULT_CLINICAL_COVARIATES
    )
    voc_metrics = analyze_voc_associations(
        analysis_frame, precombat, DEFAULT_CLINICAL_COVARIATES
    )
    cfd_metrics = analyze_covariate_associations(analysis_frame, DEFAULT_CLINICAL_COVARIATES)
    write_csv(cfd_metrics, args.output_dir / "cfd_pval_auroc.csv")
    note(f"Computed associations for {len(voc_metrics)} VOCs.")

    step(3, TOTAL_STEPS, "OPLS-DA modelling")
    opls_encoded = encode_covariates(
        clinical, age_scaling, fill_missing=False, age_first=False
    )
    write_csv(opls_encoded, args.output_dir / "cfd_dummy_mat.csv")
    opls_frame = voc.merge(opls_encoded, on="SampleID", how="inner")
    opls_frame["label"] = opls_frame["label"].map(lambda value: int(value == args.positive_label))
    opls_feature_ids = opls_frame.columns[2:].tolist()

    n_components = find_best_components(
        data=opls_frame.iloc[:, 2:].values,
        labels=opls_frame["label"].values,
        search_range=args.opls_search_range,
        use_opls=True,
        cv=args.opls_cv,
        random_state=args.random_state,
    )
    vip, _, _, _ = run_plsda(
        data=opls_frame.iloc[:, 2:].values,
        labels=opls_frame["label"].values,
        n_components=n_components,
        use_opls=True,
        return_model=True,
    )
    importance = pd.DataFrame({"VOC ID": opls_feature_ids, "vip": vip})
    note(f"Selected {n_components} OPLS components.")

    joint_metrics = voc_metrics.merge(importance, on="VOC ID", how="left")
    write_csv(joint_metrics, args.output_dir / "voc_metrics.csv")

    covariate_importance = importance.loc[
        ~importance["VOC ID"].isin(joint_metrics["VOC ID"])
    ].copy()
    write_csv(covariate_importance, args.output_dir / "cfd_vip.csv")

    step(4, TOTAL_STEPS, "Marker annotation and selection")
    voc_info = pd.read_excel(args.voc_info)
    marker_metrics = annotate_markers(
        joint_metrics,
        voc_info,
        fdr_threshold=args.fdr_threshold,
        vip_threshold=args.vip_threshold,
        auroc_threshold=args.auroc_threshold,
    )
    selected_markers = marker_metrics.loc[marker_metrics["marker_flag"]].copy()
    write_csv(selected_markers, args.output_dir / "marker_metrics.csv")
    note(f"Selected {len(selected_markers)} markers "
         f"(FDR < {args.fdr_threshold}, VIP > {args.vip_threshold}, AUROC > {args.auroc_threshold}).")
    note("Candidates await the manual qualitative identification review before RFECV.")

    step(5, TOTAL_STEPS, "RFECV feature selection")
    if args.authorized_marker_table is not None:
        if args.authorized_marker_table.suffix.lower() in {".xlsx", ".xls"}:
            authorized_table = pd.read_excel(args.authorized_marker_table)
        else:
            authorized_table = pd.read_csv(args.authorized_marker_table)
        markers = load_marker_identifiers(authorized_table)
        note(
            f"Loaded {len(markers)} authorized markers from "
            f"{args.authorized_marker_table} (post-review)."
        )
    else:
        markers = load_marker_identifiers(marker_metrics)
        note(
            f"No --authorized-marker-table supplied; RFECV uses "
            f"{len(markers)} unreviewed flagged candidates."
        )
    discovery_cfd = discovery_covariates(clinical)
    external_cfd = external_covariates(external_clinical)
    postoperative = prepare_postoperative_frame(postoperative_voc)
    discovery_model = select_model_frame(voc, markers, discovery_cfd)
    external_model = select_model_frame(external_voc, markers, external_cfd)
    postoperative_model = select_model_frame(postoperative, markers)
    selector, retained_markers = run_rfecv_feature_selection(
        discovery_model,
        markers,
        cv=args.cv,
        n_jobs=args.rfe_jobs,
        random_state=args.random_state,
    )
    note(f"RFECV retained {len(retained_markers)} of {len(markers)} candidate markers.")
    selected_discovery = discovery_model[
        ["SampleID", "label", *retained_markers, "age", "gender"]
    ].copy()
    selected_external = external_model[
        ["SampleID", "label", *retained_markers, "age", "gender"]
    ].copy()
    selected_postoperative = postoperative_model[
        ["SampleID", "label", *retained_markers]
    ].copy()
    write_csv(selected_discovery, args.output_dir / "rfecv_selected_training_set.csv")
    write_csv(selected_external, args.output_dir / "rfecv_selected_external_set.csv")
    write_csv(selected_postoperative, args.output_dir / "rfecv_selected_postOp_set.csv")
    write_csv(
        pd.DataFrame({"VOC ID": retained_markers}),
        args.output_dir / "rfecv_selected_markers.csv",
    )
    final_outputs(
        [
            (
                str(args.output_dir / "rfecv_selected_training_set.csv"),
                "RFECV-selected discovery matrix; input to workflow 06",
            ),
            (
                str(args.output_dir / "rfecv_selected_external_set.csv"),
                "RFECV-selected external matrix; input to workflow 06",
            ),
            (
                str(args.output_dir / "rfecv_selected_postOp_set.csv"),
                "RFECV-selected postoperative matrix; input to workflow 06",
            ),
            (
                str(args.output_dir / "marker_metrics.csv"),
                "flagged candidate markers (pre-review list for the manual "
                "qualitative identification step)",
            ),
            (
                str(args.output_dir / "voc_metrics.csv"),
                "full per-VOC association and VIP metrics (reference)",
            ),
        ]
    )


if __name__ == "__main__":
    run(parse_args())
