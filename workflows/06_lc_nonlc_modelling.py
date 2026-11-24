"""Workflow 06: train and evaluate the LC/non-LC marker model.

Run as ``python -m workflows.06_lc_nonlc_modelling <mode> ...`` from this
repository's root. This workflow trains classification models on the
RFECV-selected feature tables produced by workflow 05 and evaluates them at a
sensitivity-anchored cutoff. Both modes are pure modelling: feature
identification (differential analysis, manual annotation review, RFECV) is
owned by workflow 05.

Subcommands
-----------
base        Train every model family (logistic, random forest, kNN,
            XGBoost, SVM) on the RFECV-selected discovery table with
            stratified k-fold cross-validation and produce fold-level
            predictions for the discovery, external, and postoperative sets
            together with per-model sensitivity-anchored cutoffs, an
            evaluation summary, and one internal/external ROC figure per
            model. This reproduces the five-family comparison of the source
            notebook; the unadjusted XGBoost model is the one used downstream
            by workflow 07 (subgroups, roc-compare, review-scenarios).
confounding Re-fit the same model family while adjusting for confounder
            covariates. Consumes the base-mode eval_batch_pred partitions to rebuild the
            discovery train/validation split.

Model hyperparameters come from ``src.modelling.DEFAULT_MODEL_PARAMETERS``,
which records the end-point values of the source notebook's parameter
search; no hyperparameter search is performed at runtime. All base-mode
families share identical cross-validation fold partitions (fixed splitter
seed) so their scores are directly comparable, matching the source notebook's
head-to-head comparison on identical folds.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.confounding import DEFAULT_ADJUSTMENTS, run_adjustment, split_discovery_partitions
from src.evaluation import (
    METRIC_ORDER,
    bootstrap_auc_interval,
    evaluate_predictions,
    evaluation_summary,
    find_sensitivity_cutoff,
)
from src.io import write_csv
from src.modelling import (
    DEFAULT_MODEL_PARAMETERS,
    MODEL_REGISTRY,
    cross_validated_predictions,
    select_binary_labels,
)
from src.plotting import DEFAULT_TWO_CLASS_COLORS, draw_model_roc, roc_curve_payload, write_figure
from src.runlog import final_outputs, note, step

OUTPUT_SUBDIR = "06_lc_nonlc_modelling"

# Source-analysis comparison order.
MODEL_ORDER = ("logistic", "random_forest", "knn", "xgboost", "svm")
# Fixed seed for the CV splitter so every model family sees identical fold
# partitions; --random-state overrides it for fully reproducible reruns.
DEFAULT_CV_SEED = 42


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the model arguments shared by both modes."""
    parser.add_argument("--model-parameters", type=json.loads, default=None)
    parser.add_argument("--positive-label", default="LC")
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output"),
        help="Base output directory; results are written under {output-dir}/"
        f"{OUTPUT_SUBDIR}/<mode>/.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    base_parser = subparsers.add_parser(
        "base", help="Five-family marker-model comparison with cross-validated predictions."
    )
    add_model_arguments(base_parser)
    base_parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODEL_ORDER),
        help="Comma-separated model families to train (default: all five).",
    )
    base_parser.add_argument(
        "--training-set",
        type=Path,
        required=True,
        help="Workflow 05 rfecv_selected_training_set.csv.",
    )
    base_parser.add_argument(
        "--external-set",
        type=Path,
        required=True,
        help="Workflow 05 rfecv_selected_external_set.csv.",
    )
    base_parser.add_argument(
        "--postop-set",
        type=Path,
        required=True,
        help="Workflow 05 rfecv_selected_postOp_set.csv.",
    )
    base_parser.add_argument("--cv", type=int, default=5)
    base_parser.add_argument("--min-sensitivity", type=float, default=0.95)
    base_parser.add_argument("--n-bootstrap", type=int, default=1000)
    base_parser.add_argument(
        "--eval-fold",
        type=int,
        default=2,
        help="KFID of the held-out fold partition exported for evaluation "
        "(the source XGBoost analysis used fold 2).",
    )
    base_parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip ROC figure generation (requires plotly and a kaleido backend).",
    )

    confounding_parser = subparsers.add_parser(
        "confounding", help="Confounder-adjusted re-fits of the marker model."
    )
    add_model_arguments(confounding_parser)
    confounding_parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="xgboost")
    confounding_parser.add_argument(
        "--training-set", type=Path, required=True, help="Workflow 05 rfecv_selected_training_set.csv."
    )
    confounding_parser.add_argument(
        "--external-set", type=Path, required=True, help="Workflow 05 rfecv_selected_external_set.csv."
    )
    confounding_parser.add_argument(
        "--train-pred",
        type=Path,
        required=True,
        help="Base-mode eval_batch_pred training partition used to rebuild the discovery split.",
    )
    confounding_parser.add_argument(
        "--val-pred",
        type=Path,
        required=True,
        help="Base-mode eval_batch_pred validation partition used to rebuild the discovery split.",
    )
    confounding_parser.add_argument(
        "--discovery-cfd", type=Path, required=True, help="Workflow 05 cfd_dummy_mat_discovery.csv."
    )
    confounding_parser.add_argument(
        "--external-cfd", type=Path, required=True, help="Workflow 05 cfd_dummy_mat_external.csv."
    )
    confounding_parser.add_argument("--min-sensitivity", type=float, default=0.95)
    confounding_parser.add_argument("--cutoff-steps", type=int, default=100)
    confounding_parser.add_argument("--bootstrap-rounds", type=int, default=1000)
    confounding_parser.add_argument(
        "--adjustments",
        type=str,
        default=None,
        help="Comma-separated adjustment names to run (default: all).",
    )
    confounding_parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip ROC figure generation (requires plotly and a kaleido backend).",
    )

    args = parser.parse_args()
    args.output_dir = args.output_dir / OUTPUT_SUBDIR / args.mode
    return args


def run_base(args: argparse.Namespace) -> None:
    """Train every requested model family and evaluate it at its own cutoff."""
    TOTAL_STEPS = 3

    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in requested if name not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(unknown)}")

    step(1, TOTAL_STEPS, "Load RFECV-selected model frames")
    training_set = pd.read_csv(args.training_set)
    external_set = pd.read_csv(args.external_set)
    postop_set = pd.read_csv(args.postop_set)
    n_features = len(training_set.columns) - 4  # SampleID, label, age, gender
    note(
        f"Discovery: {len(training_set)} samples; external: {len(external_set)}; "
        f"postoperative: {len(postop_set)}; features: {n_features}."
    )

    model_frame_discovery = training_set.drop(columns=["age", "gender"]).copy()
    model_frame_external = external_set.drop(columns=["age", "gender"]).copy()

    # Fixed splitter seed: all model families share identical fold partitions,
    # matching the source notebook's head-to-head comparison on identical folds.
    cv_seed = args.random_state if args.random_state is not None else DEFAULT_CV_SEED

    step(2, TOTAL_STEPS, f"Cross-validated training and prediction ({len(requested)} models)")
    note(f"CV splitter seed {cv_seed}: all model families share identical fold partitions.")

    figs_dir = args.output_dir / "figs"
    summary_rows: list[pd.DataFrame] = []

    for model_name in tqdm(requested, desc="Models"):
        parameters = DEFAULT_MODEL_PARAMETERS.get(model_name, {}).copy()
        if args.model_parameters:
            parameters.update(args.model_parameters)

        train_folds, validation_folds, independent = cross_validated_predictions(
            MODEL_REGISTRY[model_name],
            parameters,
            model_frame_discovery,
            cv=args.cv,
            random_state=cv_seed,
            positive_label=args.positive_label,
            independent_frames=[model_frame_external, postop_set],
        )
        external_folds = independent[0]
        postoperative_folds = independent[1]

        write_csv(train_folds, args.output_dir / f"{model_name}_train_pred_proba.csv")
        write_csv(validation_folds, args.output_dir / f"{model_name}_val_pred_proba.csv")
        write_csv(external_folds, args.output_dir / f"{model_name}_ext_pred_proba.csv")
        write_csv(postoperative_folds, args.output_dir / f"{model_name}_postOp_pred_proba.csv")

        # The source analysis evaluates one held-out fold partition
        # (``KFID == eval_fold``) rather than fold-averaged scores. Exported
        # tables keep the original string labels; evaluation uses binarized
        # copies in memory.
        train_eval = train_folds.loc[train_folds["KFID"] == args.eval_fold].copy()
        validation_eval = validation_folds.loc[validation_folds["KFID"] == args.eval_fold].copy()
        external_eval = external_folds.loc[external_folds["KFID"] == args.eval_fold].copy()
        postoperative_eval = postoperative_folds.loc[
            postoperative_folds["KFID"] == args.eval_fold
        ].copy()
        write_csv(train_eval, args.output_dir / "eval_batch_pred" / f"{model_name}_train.csv")
        write_csv(validation_eval, args.output_dir / "eval_batch_pred" / f"{model_name}_val.csv")
        write_csv(external_eval, args.output_dir / "eval_batch_pred" / f"{model_name}_ext.csv")
        write_csv(postoperative_eval, args.output_dir / "eval_batch_pred" / f"{model_name}_postOp.csv")

        binary_frames = {
            "train": select_binary_labels(train_eval, positive_label=args.positive_label),
            "validation": select_binary_labels(validation_eval, positive_label=args.positive_label),
            "external": select_binary_labels(external_eval, positive_label=args.positive_label),
            "postoperative": select_binary_labels(postoperative_eval, positive_label=args.positive_label),
        }

        cutoff = find_sensitivity_cutoff(
            binary_frames["validation"],
            min_sensitivity=args.min_sensitivity,
            steps=100,
        )

        results: dict[str, dict[str, object]] = {}
        for dataset, frame in binary_frames.items():
            result = evaluate_predictions(frame, cutoff)
            auc_lower, auc_upper = bootstrap_auc_interval(
                frame["label"].to_numpy(dtype=int),
                frame["pred"].to_numpy(dtype=float),
                n_bootstrap=args.n_bootstrap,
                random_state=args.random_state,
            )[1:]
            result["auc_lb"] = auc_lower
            result["auc_ub"] = auc_upper
            results[dataset] = result
            summary_rows.append(
                evaluation_summary(result, model=model_name, dataset=dataset, cutoff=cutoff)
            )

        note(
            f"{model_name}: cutoff {cutoff:.4f}; "
            f"valid AUC {results['validation']['AUROC']:.3f} "
            f"({results['validation']['auc_lb']:.3f}, {results['validation']['auc_ub']:.3f}); "
            f"external AUC {results['external']['AUROC']:.3f} "
            f"({results['external']['auc_lb']:.3f}, {results['external']['auc_ub']:.3f})"
        )

        if not args.skip_figures:
            figure = draw_model_roc(
                roc_curve_payload(results["validation"], "Internal Test Set"),
                roc_curve_payload(results["external"], "External Test Set"),
                color_scheme=DEFAULT_TWO_CLASS_COLORS,
                auc_bounds={
                    "Internal Test Set": (
                        results["validation"]["auc_lb"],
                        results["validation"]["auc_ub"],
                    ),
                    "External Test Set": (
                        results["external"]["auc_lb"],
                        results["external"]["auc_ub"],
                    ),
                },
            )
            write_figure(figure, figs_dir / f"{model_name}_roc.png")

    step(3, TOTAL_STEPS, "Summarize evaluations")
    write_csv(pd.concat(summary_rows, ignore_index=True), args.output_dir / "evaluation_summary.csv")
    note(f"Wrote evaluation_summary.csv ({len(summary_rows)} model x dataset evaluations).")
    final_outputs(
        [
            (
                str(args.output_dir / "evaluation_summary.csv"),
                "AUROC and metrics per model family and dataset at each sensitivity-anchored cutoff",
            ),
            (
                str(args.output_dir / "eval_batch_pred"),
                "held-out fold partitions per model; xgboost_val/ext/postOp are the inputs to workflow 07 (subgroups, roc-compare, review-scenarios)",
            ),
            (
                str(figs_dir),
                "one internal/external ROC figure per model family ({model}_roc.png)",
            ),
        ]
    )


def run_confounding(args: argparse.Namespace) -> None:
    """Fit every confounding-adjustment model and summarize its performance."""
    TOTAL_STEPS = 3

    step(1, TOTAL_STEPS, "Load inputs and split discovery partitions")
    training_set = select_binary_labels(
        pd.read_csv(args.training_set), positive_label=args.positive_label
    )
    external_set = select_binary_labels(
        pd.read_csv(args.external_set), positive_label=args.positive_label
    )
    train_predictions = pd.read_csv(args.train_pred)
    validation_predictions = pd.read_csv(args.val_pred)
    discovery_confounders = pd.read_csv(args.discovery_cfd)
    external_confounders = pd.read_csv(args.external_cfd)

    train_frame, valid_frame = split_discovery_partitions(
        training_set, train_predictions, validation_predictions
    )
    note(
        f"Discovery partitions: {len(train_frame)} training / {len(valid_frame)} validation; "
        f"external: {len(external_set)}."
    )

    parameters = DEFAULT_MODEL_PARAMETERS.get(args.model, {}).copy()
    if args.model_parameters:
        parameters.update(args.model_parameters)

    data_dir = args.output_dir / "data"
    figs_dir = args.output_dir / "figs"
    summary_rows: list[pd.DataFrame] = []

    if args.adjustments:
        requested = [name.strip() for name in args.adjustments.split(",") if name.strip()]
        known = {spec.name: spec for spec in DEFAULT_ADJUSTMENTS}
        unknown = [name for name in requested if name not in known]
        if unknown:
            raise ValueError(f"Unknown adjustments: {', '.join(unknown)}")
        specs = [known[name] for name in requested]
    else:
        specs = list(DEFAULT_ADJUSTMENTS)

    step(2, TOTAL_STEPS, f"Fit adjustment models ({args.model})")
    for spec in tqdm(specs, desc="Adjustments"):
        outcome = run_adjustment(
            spec,
            MODEL_REGISTRY[args.model],
            parameters,
            train_frame,
            valid_frame,
            external_set,
            discovery_confounders,
            external_confounders,
            min_sensitivity=args.min_sensitivity,
            cutoff_steps=args.cutoff_steps,
            bootstrap_rounds=args.bootstrap_rounds,
            random_state=args.random_state,
        )
        name = outcome["adjustment"]
        note(
            f"{name}: cutoff {outcome['cutoff']:.4f}; "
            f"valid AUC {outcome['evaluations']['valid']['AUROC']:.3f}; "
            f"external AUC {outcome['evaluations']['external']['AUROC']:.3f}"
        )
        write_csv(outcome["predictions"]["train"], data_dir / f"{name}_train_pred.csv")
        write_csv(outcome["predictions"]["valid"], data_dir / f"{name}_val_pred.csv")
        write_csv(outcome["predictions"]["external"], data_dir / f"{name}_ext_pred.csv")

        for partition, dataset in [("valid", "Internal Test Set"), ("external", "External Test Set")]:
            result = outcome["evaluations"][partition]
            row: dict[str, object] = {
                "adjustment": name,
                "dataset": dataset,
                "cutoff": outcome["cutoff"],
                "auc_lb": result["auc_lb"],
                "auc_ub": result["auc_ub"],
            }
            for metric in METRIC_ORDER:
                row[metric] = result[metric]
                lower, upper = result["intervals"][metric]
                row[f"{metric}_lower"] = lower
                row[f"{metric}_upper"] = upper
            summary_rows.append(pd.DataFrame([row]))

        if not args.skip_figures:
            internal_payload = roc_curve_payload(
                outcome["evaluations"]["valid"], "Internal Test Set"
            )
            external_payload = roc_curve_payload(
                outcome["evaluations"]["external"], "External Test Set"
            )
            figure = draw_model_roc(
                internal_payload,
                external_payload,
                color_scheme=DEFAULT_TWO_CLASS_COLORS,
                auc_bounds={
                    "Internal Test Set": (
                        outcome["evaluations"]["valid"]["auc_lb"],
                        outcome["evaluations"]["valid"]["auc_ub"],
                    ),
                    "External Test Set": (
                        outcome["evaluations"]["external"]["auc_lb"],
                        outcome["evaluations"]["external"]["auc_ub"],
                    ),
                },
            )
            write_figure(figure, figs_dir / f"{args.model}_{name}_roc.png")

    step(3, TOTAL_STEPS, "Summarize evaluations")
    write_csv(pd.concat(summary_rows, ignore_index=True), args.output_dir / "evaluation_summary.csv")
    note(f"Summarized {len(summary_rows)} adjustment x dataset evaluations.")
    final_outputs(
        [
            (
                str(args.output_dir / "evaluation_summary.csv"),
                "cutoff and metric intervals per adjustment model and dataset",
            ),
            (
                str(args.output_dir / "data"),
                "per-adjustment prediction tables ({adjustment}_train/val/ext_pred.csv)",
            ),
        ]
    )


def run(args: argparse.Namespace) -> None:
    if args.mode == "base":
        run_base(args)
    else:
        run_confounding(args)


if __name__ == "__main__":
    run(parse_args())
