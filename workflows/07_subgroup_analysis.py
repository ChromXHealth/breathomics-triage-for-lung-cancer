"""Workflow 07: subgroup and post-hoc analyses of the LC/non-LC model.

Run as ``python -m workflows.07_subgroup_analysis <subcommand> ...`` from
this repository's root after workflow 06. Every subcommand consumes the
workflow 06 base-mode prediction tables (or the workflow 05 feature tables)
plus caller-supplied study-data tables; nothing here modifies predictions,
labels, or cohort assignments.

Subcommands
-----------
subgroups          Subgroup analysis and decision-curve net benefit on the
                   validation/external/(optional) training predictions.
shap               XGBoost SHAP explanation refit on the workflow 05 feature
                   tables.
pr-calibration     Precision-recall and calibration curves on validation and
                   external predictions (source notebook 13 section 4).
npv-ppv-adjust     Prevalence-adjusted PPV/NPV with confidence intervals
                   (source notebook 13 section 2).
roc-compare        Multi-model ROC comparison with DeLong tests (source
                   notebook 13 sections 8, 10-12).
confounder-models  Five-classifier modelling on the ten confounder
                   covariates with cross-validation plus an independent
                   external set (source notebook 13 section 9).
review-scenarios   Cell revision-response statistics: sampling-site label
                   counts, external-set scenario performance, and AJCC
                   stage-I scenarios.
                   The collection-to-analysis interval analysis was removed:
                   the lab-records table no longer needs
                   sampling_time/analysis_time.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.calibration import compute_brier_score, compute_calibration_curve
from src.evaluation import (
    adjust_ppv_npv,
    bootstrap_auc_interval,
    delong_test,
    evaluate_predictions,
    find_sensitivity_cutoff,
    precision_recall_with_interval,
    require_columns,
)
from src.io import write_csv
from src.modelling import (
    DEFAULT_MODEL_PARAMETERS,
    MODEL_REGISTRY,
    cross_validated_predictions,
    select_binary_labels,
)
from src.plotting import (
    DEFAULT_TWO_CLASS_COLORS,
    draw_boxplot,
    draw_calibration_curve,
    draw_dca_curve,
    draw_mean_abs_shap_bar,
    draw_metric_bars,
    draw_model_prc,
    draw_model_roc,
    pr_curve_payload,
    roc_curve_payload,
    write_figure,
)
from src.runlog import final_outputs, note, step
from src.subgroups import (
    DEFAULT_SUBGROUP_SPECS,
    attach_clinical_covariates,
    attach_imaging_labels,
    attach_nonlc_subtypes,
    build_vis_labels,
    compute_net_benefit,
    evaluate_subgroup,
    resolve_groups,
)

DATASET_PREDICTION_ARGS = ("val", "ext", "train")
SCENARIO_COLUMN = "scenario"
OUTPUT_SUBDIR = "07_subgroup_analysis"

CONFOUNDER_MODEL_PARAMETERS: dict[str, dict[str, object]] = {
    "logistic": {"C": 10, "penalty": "l2"},
    "random_forest": {
        "oob_score": True,
        "bootstrap": True,
        "n_estimators": 140,
        "max_depth": 9,
        "min_samples_leaf": 4,
        "min_samples_split": 9,
        "max_features": 8,
    },
    "knn": {"n_neighbors": 35, "weights": "uniform", "p": 1, "leaf_size": 10},
    "svm": {"C": 170, "gamma": 0.01, "kernel": "rbf", "class_weight": None, "probability": True},
}
CONFOUNDER_XGBOOST_PARAMETERS: dict[str, object] = {
    "colsample_bytree": 0.8,
    "gamma": 0.2,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 1,
    "n_estimators": 300,
    "reg_alpha": 1,
    "reg_lambda": 2,
    "subsample": 0.8,
}
CONFOUNDER_MODEL_ORDER = ("logistic", "random_forest", "knn", "xgboost", "svm")
CONFOUNDER_COVARIATE_COLUMNS = (
    "gender",
    "age",
    "BMI",
    "smoke_hist",
    "second-hand_smoke_status",
    "env_smoke_status",
    "hazardous_substance_exposure",
    "family_Ca_hist",
    "lung_disease_hist",
    "Ca_hist",
)


# ---------------------------------------------------------------------------
# subgroups
# ---------------------------------------------------------------------------

def load_dataset_frames(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Load and annotate every prediction frame with immutable source fields."""
    imaging = pd.read_csv(args.imaging_labels)
    clinical = pd.concat(
        [pd.read_csv(args.clinical_discovery), pd.read_csv(args.clinical_external)],
        ignore_index=True,
    )
    subtypes = pd.read_csv(args.nonlc_subtypes) if args.nonlc_subtypes else None

    frames: dict[str, pd.DataFrame] = {}
    sources = {"val": args.val_pred, "ext": args.ext_pred}
    if args.train_pred:
        sources["train"] = args.train_pred
    for dataset, path in sources.items():
        frame = pd.read_csv(path)
        require_columns(frame, ["SampleID", "label", "pred"])
        # Subgroup evaluation and figures expect an explicit 0/1 label
        # (workflow 06 prediction tables store "LC"/"nonLC" strings).
        frame = select_binary_labels(frame)
        frame = attach_imaging_labels(frame, imaging)
        frame = attach_clinical_covariates(frame, clinical)
        if subtypes is not None:
            frame = attach_nonlc_subtypes(frame, subtypes)
        frames[dataset] = frame
    return frames


def resolve_cutoff(args: argparse.Namespace) -> float:
    """Return ``--cutoff`` or the model's searched cutoff from workflow 06.

    The operating cutoff is the sensitivity-anchored value workflow 06
    searched on the validation folds and recorded in the ``cutoff`` column
    of ``evaluation_summary.csv``; callers may still override it explicitly
    via ``--cutoff``.
    """
    if args.cutoff is not None:
        return args.cutoff
    summary = pd.read_csv(args.evaluation_summary)
    require_columns(summary, ["model", "dataset", "cutoff"])
    rows = summary.loc[(summary["model"] == args.model) & summary["cutoff"].notna()]
    if rows.empty:
        raise ValueError(
            f"No cutoff row for model '{args.model}' in {args.evaluation_summary}; "
            "pass --cutoff explicitly."
        )
    validation = rows.loc[rows["dataset"] == "validation"]
    source = validation if len(validation) else rows
    cutoff = float(source["cutoff"].iloc[0])
    note(
        f"Resolved cutoff {cutoff:.4f} for model '{args.model}' from "
        f"{args.evaluation_summary} (override with --cutoff)."
    )
    return cutoff


def run_subgroups(args: argparse.Namespace) -> None:
    """Evaluate every subgroup dimension on every supplied prediction set."""
    TOTAL_STEPS = 3

    step(1, TOTAL_STEPS, "Load and annotate prediction sets")
    cutoff = resolve_cutoff(args)
    frames = load_dataset_frames(args)
    note(f"Loaded {len(frames)} prediction set(s): {', '.join(frames)}.")
    data_dir = args.output_dir / "data"
    figs_dir = args.output_dir / "figs"

    step(2, TOTAL_STEPS, "Subgroup evaluation")
    rows: list[dict] = []
    jobs = [
        (dataset, spec, group)
        for dataset, frame in frames.items()
        for spec in DEFAULT_SUBGROUP_SPECS
        if spec.source_column in frame.columns
        for group in resolve_groups(spec, frame)
    ]
    for dataset, spec, group in tqdm(jobs, desc="Subgroups"):
        frame = frames[dataset]
        row = evaluate_subgroup(frame, spec, group, cutoff, dataset)
        if row is not None:
            rows.append(row)
    metrics = pd.DataFrame(rows)
    write_csv(metrics, data_dir / "subgroup_metrics.csv")
    note(f"Evaluated {metrics['subgroup'].nunique()} subgroup dimensions "
         f"across {metrics['dataset'].nunique()} dataset(s).")

    step(3, TOTAL_STEPS, "Decision-curve analysis (tables always; figures only with --figures)")
    dca_curves = []
    for dataset in DATASET_PREDICTION_ARGS:
        if dataset not in frames:
            continue
        frame = frames[dataset][["label", "pred"]]
        net_benefit, treat_all, thresholds = compute_net_benefit(frame, steps=args.dca_steps)
        write_csv(
            pd.DataFrame(
                {
                    "threshold": thresholds,
                    "net_benefit": net_benefit,
                    "treat_all": treat_all,
                    "dataset": dataset,
                }
            ),
            data_dir / f"dca_{dataset}.csv",
        )
        dca_curves.append((f"{dataset} model", net_benefit, thresholds))

    if args.figures:
        for dataset, frame in frames.items():
            for spec in DEFAULT_SUBGROUP_SPECS:
                if spec.source_column not in frame.columns:
                    continue
                groups = [group.label for group in resolve_groups(spec, frame)]
                if spec.kind == "lc_stage":
                    vis_label = build_vis_labels(frame, spec)
                    plot_frame = frame.assign(vis_label=vis_label)
                    order = [label for label in ["nonLC", *[g.label for g in resolve_groups(spec, frame)]] if (plot_frame["vis_label"] == label).any()]
                    if order:
                        write_figure(
                            draw_boxplot(plot_frame, "vis_label", order),
                            figs_dir / f"{dataset}_{spec.name}_box.png",
                        )
                spec_rows = metrics.loc[
                    (metrics["subgroup"] == spec.name) & (metrics["dataset"] == dataset)
                ]
                if len(spec_rows):
                    write_figure(
                        draw_metric_bars(spec_rows.to_dict("records"), metric="Sensitivity"),
                        figs_dir / f"{dataset}_{spec.name}_sensitivity.png",
                    )
        if dca_curves:
            write_figure(draw_dca_curve(dca_curves), figs_dir / "dca_curve.png")

    final_outputs(
        [
            (
                str(data_dir / "subgroup_metrics.csv"),
                "sensitivity/specificity/PPV/NPV per subgroup, dataset, and cutoff",
            ),
            (
                str(figs_dir),
                "subgroup figures and dca_curve.png (written only with --figures)",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# shap
# ---------------------------------------------------------------------------

def _load_shap() -> object:
    """Import shap lazily so CLI help works without it installed."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is required for model explanation. Install it via the "
            "project environment (see environment.yml)."
        ) from exc
    return shap


def _load_xgboost() -> object:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for model explanation. Install it via the "
            "project environment (see environment.yml)."
        ) from exc
    return XGBClassifier


def _features_and_labels(frame: pd.DataFrame, positive_label: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Split a model matrix into features, binary labels, and feature names."""
    require_columns(frame, ["SampleID", "label"])
    feature_names = frame.columns[2:].tolist()
    features = frame[feature_names].to_numpy(dtype=float)
    labels = frame["label"].map(lambda value: int(value == positive_label)).astype(int).to_numpy()
    return features, labels, feature_names


def run_shap(args: argparse.Namespace) -> None:
    """Fit the manuscript model and compute SHAP explanations for both sets."""
    TOTAL_STEPS = 3
    import json

    step(1, TOTAL_STEPS, "Load model matrices and fit XGBoost")
    shap = _load_shap()
    xgboost_classifier = _load_xgboost()

    training_frame = select_binary_labels(
        pd.read_csv(args.training_set), positive_label=args.positive_label
    )
    external_frame = select_binary_labels(
        pd.read_csv(args.external_set), positive_label=args.positive_label
    )

    train_features, train_labels, feature_names = _features_and_labels(
        training_frame, args.positive_label
    )
    ext_features, _, _ = _features_and_labels(external_frame, args.positive_label)

    parameters = dict(DEFAULT_MODEL_PARAMETERS["xgboost"])
    if args.model_parameters:
        parameters.update(json.loads(args.model_parameters))
    parameters.setdefault("random_state", args.random_state)

    model = xgboost_classifier(**parameters)
    model.fit(train_features, train_labels)
    note(f"Fit on {len(train_features)} training samples with {len(feature_names)} features.")

    step(2, TOTAL_STEPS, "Compute SHAP values")
    explainer = shap.TreeExplainer(model)
    train_shap = explainer.shap_values(train_features)
    ext_shap = explainer.shap_values(ext_features)
    note(f"Explained {len(train_features)} training and {len(ext_features)} external samples.")

    step(3, TOTAL_STEPS, "Export tables and figures")
    mean_abs = pd.DataFrame(
        {
            "feature": feature_names,
            "internal_mean_abs_shap": np.abs(train_shap).mean(axis=0),
            "external_mean_abs_shap": np.abs(ext_shap).mean(axis=0),
        }
    ).sort_values("external_mean_abs_shap", ascending=False)
    write_csv(mean_abs, args.output_dir / "data" / "mean_abs_shap.csv")
    note(f"Ranked {len(mean_abs)} features by mean |SHAP|.")

    if args.figures:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for the beeswarm figure. Install it "
                "via the project environment (see environment.yml)."
            ) from exc
        figure_dir = args.output_dir / "figures"
        bar_fig = draw_mean_abs_shap_bar(
            mean_abs,
            feature_column="feature",
            value_columns=("internal_mean_abs_shap", "external_mean_abs_shap"),
        )
        write_figure(bar_fig, figure_dir / "mean_abs_shap_bar.png")

        beeswarm_path = figure_dir / "shap_beeswarm_external.png"
        shap.summary_plot(
            ext_shap,
            ext_features,
            feature_names=feature_names,
            max_display=args.max_display,
            show=False,
        )
        figure = plt.gcf()
        figure.savefig(str(beeswarm_path), dpi=300, bbox_inches="tight")
        plt.close(figure)

    final_outputs(
        [
            (
                str(args.output_dir / "data" / "mean_abs_shap.csv"),
                "mean |SHAP| per feature on training and external sets",
            ),
            (
                str(args.output_dir / "figures"),
                "mean-|SHAP| bar plot and external beeswarm (written only with --figures)",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# review-response analyses
# ---------------------------------------------------------------------------

def run_pr_calibration(args: argparse.Namespace) -> None:
    """Compute PR and calibration statistics and figures for both datasets."""
    TOTAL_STEPS = 2
    step(1, TOTAL_STEPS, "Compute PR and calibration statistics")
    figure_dir = args.output_dir / "figures"
    rows: list[dict[str, object]] = []
    payloads: dict[str, list[dict[str, object]]] = {"pr": [], "calibration": []}

    for dataset, path in tqdm(
        (("validation", args.val_pred), ("external", args.ext_pred)), desc="Datasets"
    ):
        frame = pd.read_csv(path)
        require_columns(frame, ["SampleID", "label", "pred"])
        frame = select_binary_labels(frame)
        labels = frame["label"].to_numpy(dtype=int)
        scores = frame["pred"].to_numpy(dtype=float)

        pr = precision_recall_with_interval(
            labels, scores, n_bootstrap=args.n_bootstrap, random_state=args.random_state
        )
        calibration = compute_calibration_curve(
            labels,
            scores,
            n_bins=args.n_bins,
            return_ci=True,
            n_bootstrap=args.n_bootstrap,
        )
        brier = compute_brier_score(
            labels, scores, return_ci=True, n_bootstrap=args.n_bootstrap
        )

        rows.append(
            {
                "dataset": dataset,
                "n": int(len(frame)),
                "prevalence": float(labels.mean()),
                "ap": pr["ap"],
                "ap_lower": pr["ap_lower"],
                "ap_upper": pr["ap_upper"],
                "brier": brier["brier_score"],
                "brier_lower": brier["brier_ci_lower"],
                "brier_upper": brier["brier_ci_upper"],
                "reliability": brier["reliability"],
                "resolution": brier["resolution"],
                "uncertainty": brier["uncertainty"],
            }
        )
        payloads["pr"].append(pr_curve_payload(pr, dataset))
        payloads["calibration"].append((dataset, calibration, brier))

        pr_points = pd.DataFrame(
            {"dataset": dataset, "recall": pr["recall"], "precision": pr["precision"]}
        )
        write_csv(pr_points, args.output_dir / "data" / f"pr_curve_{dataset}.csv")
        calibration_points = pd.DataFrame(
            {
                "dataset": dataset,
                "bin_center": calibration["bin_centers"],
                "fraction_positive": calibration["fraction_positives"],
                "count": calibration["counts"],
                "ci_lower": calibration.get("ci_lower", np.full_like(calibration["bin_centers"], np.nan)),
                "ci_upper": calibration.get("ci_upper", np.full_like(calibration["bin_centers"], np.nan)),
            }
        )
        write_csv(
            calibration_points, args.output_dir / "data" / f"calibration_curve_{dataset}.csv"
        )

    step(2, TOTAL_STEPS, "Export summary and figures")
    write_csv(pd.DataFrame(rows), args.output_dir / "data" / "pr_calibration_summary.csv")
    note(f"Summarized {len(rows)} dataset(s) (AP, Brier, reliability).")

    if args.figures:
        colors = [DEFAULT_TWO_CLASS_COLORS[i % 2] for i in range(len(payloads["pr"]))]
        pr_fig = draw_model_prc(*payloads["pr"], color_scheme=colors)
        write_figure(pr_fig, figure_dir / "pr_curves.png")
        for dataset, calibration, brier in payloads["calibration"]:
            cal_fig = draw_calibration_curve(calibration, model_name=dataset, brier=brier)
            write_figure(cal_fig, figure_dir / f"calibration_curve_{dataset}.png")
    final_outputs(
        [
            (
                str(args.output_dir / "data" / "pr_calibration_summary.csv"),
                "AP, Brier score, and reliability per dataset",
            ),
            (
                str(args.output_dir / "figures"),
                "PR and calibration curves (written only with --figures)",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# npv-ppv-adjust
# ---------------------------------------------------------------------------


def run_npv_ppv_adjust(args: argparse.Namespace) -> None:
    """Adjust PPV/NPV for each requested target prevalence."""
    step(1, 1, "Prevalence-adjust PPV/NPV")
    rows = []
    for prevalence in tqdm(args.prevalences, desc="Prevalences"):
        result = adjust_ppv_npv(
            prevalence=prevalence,
            sensitivity=args.sensitivity,
            specificity=args.specificity,
            n_sensitivity=args.n_sensitivity,
            n_specificity=args.n_specificity,
            n_prevalence=args.n_prevalence,
            method=args.method,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
        )
        rows.append({key: value for key, value in result.items() if key != "confidence"})
    output_path = args.output_dir / "data" / "prevalence_adjusted_ppv_npv.csv"
    write_csv(pd.DataFrame(rows), output_path)
    final_outputs([(str(output_path), "prevalence-adjusted PPV/NPV with intervals")])


# ---------------------------------------------------------------------------
# roc-compare
# ---------------------------------------------------------------------------


def parse_model_specs(specs: list[str]) -> dict[str, str]:
    """Parse repeated ``NAME=COLUMN`` flags into an ordered name->column map."""
    mapping: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--model must be NAME=COLUMN, got: {spec}")
        name, column = spec.split("=", 1)
        name = name.strip()
        column = column.strip()
        if not name or not column:
            raise ValueError(f"--model must be NAME=COLUMN, got: {spec}")
        if name in mapping:
            raise ValueError(f"Duplicate model name: {name}")
        mapping[name] = column
    return mapping


def run_roc_compare(args: argparse.Namespace) -> None:
    """Evaluate every model, draw the joint ROC, and run pairwise DeLong tests."""
    TOTAL_STEPS = 2
    step(1, TOTAL_STEPS, "Evaluate models and bootstrap AUROC intervals")
    frame = pd.read_csv(args.pred)
    model_columns = parse_model_specs(args.models)
    require_columns(frame, ["label", *model_columns.values()])
    if not pd.api.types.is_integer_dtype(frame["label"]):
        # Tolerate "LC"/"nonLC" string labels in caller-prepared tables.
        frame = select_binary_labels(frame)
    labels = frame["label"].to_numpy(dtype=int)

    figure_dir = args.output_dir / "figures"
    metrics_rows: list[dict[str, object]] = []
    roc_payloads: list[dict[str, object]] = []
    auc_bounds: dict[str, tuple[float, float]] = {}
    scores_by_model: dict[str, np.ndarray] = {}

    for index, (name, column) in enumerate(
        tqdm(list(model_columns.items()), desc="Models")
    ):
        scores = frame[column].to_numpy(dtype=float)
        scores_by_model[name] = scores
        prediction_frame = pd.DataFrame({"label": labels, "pred": scores})
        auc, auc_lower, auc_upper = bootstrap_auc_interval(
            labels,
            scores,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
        )
        note(f"{name}: AUROC {auc:.3f} ({auc_lower:.3f}-{auc_upper:.3f})")
        row: dict[str, object] = {
            "model": name,
            "n": int(len(frame)),
            "AUROC": auc,
            "AUROC_lower": auc_lower,
            "AUROC_upper": auc_upper,
        }
        if args.cutoff is not None:
            result = evaluate_predictions(prediction_frame, cutoff=args.cutoff)
            for metric in ("Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1"):
                row[metric] = result[metric]
        metrics_rows.append(row)
        roc_payloads.append(roc_curve_payload({"AUROC": auc, "roc": _roc_arrays(labels, scores)}, name))
        auc_bounds[name] = (auc_lower, auc_upper)

        curve = pd.DataFrame(_roc_arrays(labels, scores))
        write_csv(curve.assign(model=name), args.output_dir / "data" / f"roc_curve_{index}_{name}.csv")

    step(2, TOTAL_STEPS, "Pairwise DeLong tests and export")
    delong_rows: list[dict[str, object]] = []
    names = list(model_columns)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            test = delong_test(labels, scores_by_model[names[i]], scores_by_model[names[j]])
            delong_rows.append(
                {
                    "model1": names[i],
                    "model2": names[j],
                    "auc1": test["auc_first"],
                    "auc2": test["auc_second"],
                    "auc_difference": test["auc_difference"],
                    "z_statistic": test["z_statistic"],
                    "p_value": test["p_value"],
                    "n_positive": test["n_positive"],
                    "n_negative": test["n_negative"],
                }
            )
    note(f"Ran {len(delong_rows)} pairwise DeLong test(s).")

    write_csv(pd.DataFrame(metrics_rows), args.output_dir / "data" / "roc_metrics.csv")
    write_csv(pd.DataFrame(delong_rows), args.output_dir / "data" / "delong_tests.csv")

    if args.figures:
        colors = [DEFAULT_TWO_CLASS_COLORS[i % 2] for i in range(len(roc_payloads))]
        fig = draw_model_roc(*roc_payloads, color_scheme=colors, auc_bounds=auc_bounds)
        write_figure(fig, figure_dir / "roc_comparison.png")
    final_outputs(
        [
            (
                str(args.output_dir / "data" / "roc_metrics.csv"),
                "AUROC with bootstrap intervals per model (plus cutoff metrics when --cutoff is set)",
            ),
            (
                str(args.output_dir / "data" / "delong_tests.csv"),
                "pairwise DeLong test results",
            ),
            (
                str(figure_dir / "roc_comparison.png"),
                "joint ROC figure (written only with --figures)",
            ),
        ]
    )


def _roc_arrays(labels: np.ndarray, scores: np.ndarray) -> dict[str, np.ndarray]:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    return {"fpr": fpr, "tpr": tpr}


# ---------------------------------------------------------------------------
# confounder-models
# ---------------------------------------------------------------------------


def build_confounder_frame(
    cinfo: pd.DataFrame, age_center: float | None = None, age_scale: float | None = None
) -> tuple[pd.DataFrame, float, float]:
    """Dummy-encode the ten confounder covariates and z-score age.

    The unknown-category dummies (``*_UNK``) and the ``gender_F`` reference
    column are dropped, matching the source analysis. Age is standardized
    with the discovery mean/standard deviation; the same transform is applied
    to the external set by passing the returned center and scale.
    """
    require_columns(cinfo, ["SampleID", "label", *CONFOUNDER_COVARIATE_COLUMNS])
    encoded = cinfo[["SampleID", "label"]].merge(
        pd.get_dummies(cinfo[list(CONFOUNDER_COVARIATE_COLUMNS)]),
        left_index=True,
        right_index=True,
    )
    drop_columns = [column for column in encoded.columns if column.endswith("_UNK")]
    drop_columns += [column for column in ("gender_F", "gender_Female") if column in encoded.columns]
    encoded = encoded.drop(columns=drop_columns)

    ages = encoded["age"].astype(float)
    if age_center is None or age_scale is None:
        age_center = float(ages.mean())
        age_scale = float(ages.std())
    encoded["age"] = (ages - age_center) / age_scale
    return encoded, age_center, age_scale


def run_confounder_models(args: argparse.Namespace) -> None:
    """Train each classifier family on confounders only and evaluate it."""
    TOTAL_STEPS = 2
    step(1, TOTAL_STEPS, "Encode confounder covariates")
    discovery = pd.read_csv(args.discovery_cinfo)
    external = pd.read_csv(args.external_cinfo)
    discovery_frame, age_center, age_scale = build_confounder_frame(discovery)
    external_frame, _, _ = build_confounder_frame(external, age_center, age_scale)
    note(f"Encoded {discovery_frame.shape[1] - 2} confounder features from "
         f"{len(discovery_frame)} discovery samples.")

    data_dir = args.output_dir / "data"
    figure_dir = args.output_dir / "figures"
    metric_frames: list[pd.DataFrame] = []
    roc_payloads: list[dict[str, object]] = []
    auc_bounds: dict[str, tuple[float, float]] = {}

    step(2, TOTAL_STEPS, "Train and evaluate classifier families")
    for name in tqdm(CONFOUNDER_MODEL_ORDER, desc="Models"):
        parameters = dict(CONFOUNDER_MODEL_PARAMETERS.get(name, {}))
        if name == "xgboost":
            negative = int((discovery_frame["label"] != "LC").sum())
            positive = int((discovery_frame["label"] == "LC").sum())
            parameters.update(CONFOUNDER_XGBOOST_PARAMETERS)
            parameters["scale_pos_weight"] = negative / positive if positive else 1.0

        train_pred, val_pred, independent = cross_validated_predictions(
            MODEL_REGISTRY[name],
            parameters,
            discovery_frame,
            cv=args.cv,
            random_state=args.random_state,
            independent_frames=(external_frame,),
        )
        ext_pred = independent[0]

        write_csv(train_pred, data_dir / f"{name}_train_pred_proba.csv")
        write_csv(val_pred, data_dir / f"{name}_val_pred_proba.csv")
        write_csv(ext_pred, data_dir / f"{name}_ext_pred_proba.csv")

        # Threshold search on all validation folds, evaluation on one fold,
        # mirroring the workflow 06 protocol.
        cutoff = find_sensitivity_cutoff(
            select_binary_labels(val_pred), min_sensitivity=args.min_sensitivity, steps=100
        )
        fold_val = val_pred.loc[val_pred["KFID"] == args.eval_fold]
        fold_ext = ext_pred.loc[ext_pred["KFID"] == args.eval_fold]

        for dataset, fold_frame in (("validation", fold_val), ("external", fold_ext)):
            if fold_frame.empty:
                continue
            result = evaluate_predictions(
                select_binary_labels(fold_frame), cutoff=cutoff
            )
            metric_frames.append(
                pd.DataFrame(
                    [
                        {
                            "model": name,
                            "dataset": dataset,
                            "eval_fold": args.eval_fold,
                            "cutoff": cutoff,
                            "n": int(len(fold_frame)),
                            **{metric: result[metric] for metric in ("AUROC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1")},
                        }
                    ]
                )
            )

        fold_ext_binary = select_binary_labels(fold_ext)
        labels = fold_ext_binary["label"].to_numpy(dtype=int)
        scores = fold_ext_binary["pred"].to_numpy(dtype=float)
        auc, auc_lower, auc_upper = bootstrap_auc_interval(
            labels, scores, n_bootstrap=1000, random_state=args.random_state
        )
        roc_payloads.append(
            roc_curve_payload({"AUROC": auc, "roc": _roc_arrays(labels, scores)}, name)
        )
        auc_bounds[name] = (auc_lower, auc_upper)

    write_csv(pd.concat(metric_frames, ignore_index=True), data_dir / "confounder_metrics.csv")
    note(f"Wrote metrics for {len(roc_payloads)} classifier family(ies).")

    if args.figures:
        colors = [DEFAULT_TWO_CLASS_COLORS[i % 2] for i in range(len(roc_payloads))]
        fig = draw_model_roc(*roc_payloads, color_scheme=colors, auc_bounds=auc_bounds)
        write_figure(fig, figure_dir / "confounder_roc.png")
    final_outputs(
        [
            (
                str(data_dir / "confounder_metrics.csv"),
                "operating metrics per classifier family, dataset, and eval fold",
            ),
            (
                str(figure_dir / "confounder_roc.png"),
                "joint external ROC figure (written only with --figures)",
            ),
        ]
    )


#


# ---------------------------------------------------------------------------
# review-scenarios
# ---------------------------------------------------------------------------

def site_label_counts(lab_records: pd.DataFrame) -> pd.DataFrame:
    """Count records per sampling site, dataset, and label."""
    require_columns(lab_records, ["dset", "label"])
    if "site" not in lab_records.columns:
        raise ValueError("The lab records table must contain a 'site' column.")
    counts = (
        lab_records.groupby(["dset", "label", "site"]).size().reset_index(name="n")
    )
    counts["label"] = counts["label"].map(lambda value: "LC" if value == 1 else "nonLC")
    return counts


def attach_scenario(predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach the scenario column derived from the sample-ID project prefix."""
    output = predictions.copy()
    output[SCENARIO_COLUMN] = output["SampleID"].map(lambda value: str(value).split("#")[0])
    return output


def evaluate_scenarios(predictions: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Evaluate every scenario subset at the fixed cutoff."""
    require_columns(predictions, ["SampleID", "label", "pred"])
    rows = []
    scenarios = attach_scenario(predictions).groupby(SCENARIO_COLUMN)
    for scenario, group in tqdm(scenarios, total=scenarios.ngroups, desc="Scenarios"):
        if group["label"].nunique() < 2:
            continue
        result = evaluate_predictions(group[["SampleID", "label", "pred"]], cutoff)
        row: dict[str, object] = {"scenario": scenario, "n": int(len(group))}
        for metric in ("AUROC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV"):
            row[metric] = result[metric]
            lower, upper = result["intervals"][metric]
            row[f"{metric}_lower"] = lower
            row[f"{metric}_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_ajcc_stage_scenarios(
    predictions: pd.DataFrame,
    imaging_labels: pd.DataFrame,
    cutoff: float,
) -> pd.DataFrame:
    """Evaluate AJCC stage-I LC together with every non-LC sample, per scenario."""
    require_columns(predictions, ["SampleID", "label", "pred"])
    require_columns(imaging_labels, ["SampleID", "ajcc_stage"])
    merged = predictions.merge(imaging_labels[["SampleID", "ajcc_stage"]], on="SampleID", how="left")
    stage_one = {"I", "IA", "IA1", "IA2", "IA3", "IB"}
    mask = merged["ajcc_stage"].isin(stage_one) | (merged["label"] == 0)
    targeted = attach_scenario(merged.loc[mask].copy())
    rows = []
    for scenario, group in targeted.groupby(SCENARIO_COLUMN):
        if group["label"].nunique() < 2:
            continue
        result = evaluate_predictions(group[["SampleID", "label", "pred"]], cutoff)
        row: dict[str, object] = {"scenario": scenario, "n": int(len(group))}
        for metric in ("AUROC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV"):
            row[metric] = result[metric]
            lower, upper = result["intervals"][metric]
            row[f"{metric}_lower"] = lower
            row[f"{metric}_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def run_review_scenarios(args: argparse.Namespace) -> None:
    """Compute the review-response statistics from caller-supplied records."""
    TOTAL_STEPS = 3

    cutoff = resolve_cutoff(args)

    step(1, TOTAL_STEPS, "Sampling-site label counts")
    lab_records = pd.read_csv(args.lab_records)
    require_columns(lab_records, ["SampleID", "dset", "label"])
    lab_records = select_binary_labels(lab_records)
    data_dir = args.output_dir / "data"

    site_source = pd.read_csv(args.sampling_sites) if args.sampling_sites else lab_records
    if "site" in site_source.columns:
        counts = site_label_counts(site_source)
        write_csv(counts, data_dir / "site_label_counts.csv")
        note(f"Counted {counts['site'].nunique()} sampling site(s).")
    else:
        note("No 'site' column available; skipped site label counts.")

    step(2, TOTAL_STEPS, "External-set scenario evaluation")
    ext_predictions = select_binary_labels(pd.read_csv(args.ext_pred))
    scenario_metrics = evaluate_scenarios(ext_predictions, cutoff)
    write_csv(scenario_metrics, data_dir / "scenario_metrics.csv")
    note(f"Evaluated {len(scenario_metrics)} scenario(s).")

    step(3, TOTAL_STEPS, "AJCC stage-I scenario evaluation")
    if args.imaging_labels:
        stage_one_metrics = evaluate_ajcc_stage_scenarios(
            ext_predictions, pd.read_csv(args.imaging_labels), cutoff
        )
        write_csv(stage_one_metrics, data_dir / "ajcc_stage1_scenario_metrics.csv")
        note(f"Evaluated {len(stage_one_metrics)} stage-I scenario(s).")
    else:
        note("No --imaging-labels supplied; skipped stage-I scenarios.")
    final_outputs(
        [
            (
                str(data_dir / "scenario_metrics.csv"),
                "external-set performance per scenario at the operating cutoff",
            ),
            (
                str(data_dir / "ajcc_stage1_scenario_metrics.csv"),
                "stage-I scenario performance (written only with --imaging-labels)",
            ),
            (
                str(data_dir / "site_label_counts.csv"),
                "sampling-site label counts (written only when site labels exist)",
            ),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    sub_parser = subparsers.add_parser(
        "subgroups", help="Subgroup analysis and decision-curve net benefit."
    )
    sub_parser.add_argument("--val-pred", type=Path, required=True)
    sub_parser.add_argument("--ext-pred", type=Path, required=True)
    sub_parser.add_argument("--train-pred", type=Path, default=None)
    sub_parser.add_argument("--cutoff", type=float, default=None)
    sub_parser.add_argument(
        "--evaluation-summary",
        type=Path,
        default=Path("analysis-output/06_lc_nonlc_modelling/base/evaluation_summary.csv"),
        help="Workflow 06 base evaluation_summary.csv; used to resolve --cutoff "
        "when the cutoff is not given explicitly (takes the model's validation row).",
    )
    sub_parser.add_argument(
        "--model",
        default="xgboost",
        help="Model family whose cutoff is read from --evaluation-summary "
        "(default: xgboost, the manuscript head-to-head winner).",
    )
    sub_parser.add_argument("--imaging-labels", type=Path, required=True)
    sub_parser.add_argument("--clinical-discovery", type=Path, required=True)
    sub_parser.add_argument("--clinical-external", type=Path, required=True)
    sub_parser.add_argument("--nonlc-subtypes", type=Path, default=None)
    sub_parser.add_argument("--dca-steps", type=int, default=101)
    sub_parser.add_argument(
        "--figures",
        action="store_true",
        help="Render figures (default: tables only; figure rendering is the slow step).",
    )
    sub_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output"),
        help="Base output directory; results are written under {output-dir}/"
        f"{OUTPUT_SUBDIR}/subgroups/.",
    )

    shap_parser = subparsers.add_parser("shap", help="XGBoost SHAP model explanation.")
    shap_parser.add_argument("--training-set", type=Path, required=True)
    shap_parser.add_argument("--external-set", type=Path, required=True)
    shap_parser.add_argument(
        "--model-parameters",
        type=str,
        default=None,
        help="JSON object overriding the default XGBoost parameters.",
    )
    shap_parser.add_argument("--positive-label", default="LC")
    shap_parser.add_argument(
        "--max-display", type=int, default=20, help="Features shown in the beeswarm plot."
    )
    shap_parser.add_argument("--random-state", type=int, default=None)
    shap_parser.add_argument(
        "--figures",
        action="store_true",
        help="Render figures (default: tables only; figure rendering is the slow step).",
    )
    shap_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output"),
        help="Base output directory; results are written under {output-dir}/"
        f"{OUTPUT_SUBDIR}/shap/.",
    )

    pr_parser = subparsers.add_parser(
        "pr-calibration", help="PR and calibration curves on prediction tables."
    )
    pr_parser.add_argument("--val-pred", type=Path, required=True)
    pr_parser.add_argument("--ext-pred", type=Path, required=True)
    pr_parser.add_argument("--n-bins", type=int, default=10)
    pr_parser.add_argument("--n-bootstrap", type=int, default=1000)
    pr_parser.add_argument("--random-state", type=int, default=None)
    pr_parser.add_argument(
        "--figures",
        action="store_true",
        help="Render figures (default: tables only; figure rendering is the slow step).",
    )
    pr_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output"),
        help="Base output directory; results are written under {output-dir}/"
        f"{OUTPUT_SUBDIR}/pr-calibration/.",
    )

    adjust_parser = subparsers.add_parser(
        "npv-ppv-adjust", help="Prevalence-adjusted PPV/NPV with intervals."
    )
    adjust_parser.add_argument("--sensitivity", type=float, required=True)
    adjust_parser.add_argument("--specificity", type=float, required=True)
    adjust_parser.add_argument("--prevalences", type=float, nargs="+", required=True)
    adjust_parser.add_argument("--n-sensitivity", type=int, default=None)
    adjust_parser.add_argument("--n-specificity", type=int, default=None)
    adjust_parser.add_argument("--n-prevalence", type=int, default=None)
    adjust_parser.add_argument(
        "--method", choices=("bootstrap", "wilson", "bayesian"), default="bootstrap"
    )
    adjust_parser.add_argument("--n-bootstrap", type=int, default=10000)
    adjust_parser.add_argument("--random-state", type=int, default=None)
    adjust_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output"),
        help="Base output directory; results are written under {output-dir}/"
        f"{OUTPUT_SUBDIR}/npv-ppv-adjust/.",
    )

    roc_parser = subparsers.add_parser(
        "roc-compare", help="Multi-model ROC curves and pairwise DeLong tests."
    )
    roc_parser.add_argument("--pred", type=Path, required=True)
    roc_parser.add_argument(
        "--model",
        dest="models",
        action="append",
        required=True,
        metavar="NAME=COLUMN",
        help="Model display name and its probability column; repeatable.",
    )
    roc_parser.add_argument("--cutoff", type=float, default=None)
    roc_parser.add_argument("--n-bootstrap", type=int, default=1000)
    roc_parser.add_argument("--random-state", type=int, default=None)
    roc_parser.add_argument(
        "--figures",
        action="store_true",
        help="Render figures (default: tables only; figure rendering is the slow step).",
    )
    roc_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output"),
        help="Base output directory; results are written under {output-dir}/"
        f"{OUTPUT_SUBDIR}/roc-compare/.",
    )

    conf_parser = subparsers.add_parser(
        "confounder-models",
        help="Five classifier families trained on the ten confounder covariates.",
    )
    conf_parser.add_argument("--discovery-cinfo", type=Path, required=True)
    conf_parser.add_argument("--external-cinfo", type=Path, required=True)
    conf_parser.add_argument("--eval-fold", type=int, default=2)
    conf_parser.add_argument("--min-sensitivity", type=float, default=0.95)
    conf_parser.add_argument("--cv", type=int, default=5)
    conf_parser.add_argument("--random-state", type=int, default=None)
    conf_parser.add_argument(
        "--figures",
        action="store_true",
        help="Render figures (default: tables only; figure rendering is the slow step).",
    )
    conf_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output"),
        help="Base output directory; results are written under {output-dir}/"
        f"{OUTPUT_SUBDIR}/confounder-models/.",
    )

    scen_parser = subparsers.add_parser(
        "review-scenarios", help="Cell revision-response scenario statistics."
    )
    scen_parser.add_argument(
        "--lab-records",
        type=Path,
        required=True,
        help="Joint SampleID/dset/label table with an optional 'site' column; "
        "the interval analysis was removed, so no time columns are needed.",
    )
    scen_parser.add_argument("--sampling-sites", type=Path, default=None)
    scen_parser.add_argument("--ext-pred", type=Path, required=True)
    scen_parser.add_argument("--cutoff", type=float, default=None)
    scen_parser.add_argument(
        "--evaluation-summary",
        type=Path,
        default=Path("analysis-output/06_lc_nonlc_modelling/base/evaluation_summary.csv"),
        help="Workflow 06 base evaluation_summary.csv; used to resolve --cutoff "
        "when the cutoff is not given explicitly (takes the model's validation row).",
    )
    scen_parser.add_argument(
        "--model",
        default="xgboost",
        help="Model family whose cutoff is read from --evaluation-summary "
        "(default: xgboost, the manuscript head-to-head winner).",
    )
    scen_parser.add_argument("--imaging-labels", type=Path, default=None)
    scen_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output"),
        help="Base output directory; results are written under {output-dir}/"
        f"{OUTPUT_SUBDIR}/review-scenarios/.",
    )

    args = parser.parse_args()
    if hasattr(args, "output_dir"):
        args.output_dir = args.output_dir / OUTPUT_SUBDIR / args.subcommand
    return args


def run(args: argparse.Namespace) -> None:
    handlers = {
        "subgroups": run_subgroups,
        "shap": run_shap,
        "pr-calibration": run_pr_calibration,
        "npv-ppv-adjust": run_npv_ppv_adjust,
        "roc-compare": run_roc_compare,
        "confounder-models": run_confounder_models,
        "review-scenarios": run_review_scenarios,
    }
    handlers[args.subcommand](args)


if __name__ == "__main__":
    run(parse_args())
