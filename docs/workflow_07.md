# Workflow 07: Subgroup and Post-hoc Analyses

## Purpose

Workflow 07 collects every analysis that consumes the workflow 06 model
predictions (or the workflow 05 feature tables) without changing them:
subgroup performance, decision-curve net benefit, model explanation, and the
review-response statistics. Each analysis is a subcommand of
`workflows/07_subgroup_analysis.py`; all write under
`{output-dir}/07_subgroup_analysis/<subcommand>/` unless noted.

| Subcommand | What it computes |
|---|---|---|
| `subgroups` | Subgroup sensitivity/specificity/PPV/NPV on validation/external/(optional) training predictions plus decision-curve net benefit |
| `shap` | XGBoost refit on the workflow 05 feature tables; mean \|SHAP\| per feature and beeswarm |
| `pr-calibration` | Precision-recall and calibration curves with bootstrap intervals |
| `npv-ppv-adjust` | Prevalence-adjusted PPV/NPV with intervals |
| `roc-compare` | Multi-model ROC comparison with pairwise DeLong tests |
| `confounder-models` | Five classifier families trained on the ten clinical-factor covariates |
| `review-scenarios` | Sampling-site counts, external scenario and AJCC stage-I performance |

Every subgroup statistic is derived from immutable source fields and the
fixed operating cutoff; predictions, labels, and cohort assignments are
never modified.

**Computation and figures are separated**: every subcommand writes its data
tables by default, and figure rendering (the slow step) only happens when
`--figures` is passed. A figure-only rerun never recomputes the tables'
statistics, but it does re-read the inputs.

## `subgroups`

Evaluate the LC/non-LC model across clinical and imaging subgroups and
compute decision-curve analysis (DCA) net benefit curves.

```bash
python -m workflows.07_subgroup_analysis subgroups \
  --val-pred /path/to/06_lc_nonlc_modelling/base/eval_batch_pred/xgboost_val.csv \
  --ext-pred /path/to/06_lc_nonlc_modelling/base/eval_batch_pred/xgboost_ext.csv \
  --imaging-labels /path/to/imaging_labels.csv \
  --clinical-discovery /path/to/cinfo_LC_nonLC_discovery.csv \
  --clinical-external /path/to/cinfo_LC_nonLC_external.csv
```

`--cutoff` is optional: when omitted, the operating cutoff is resolved from
the workflow 06 base `evaluation_summary.csv` (`--evaluation-summary`,
default `analysis-output/06_lc_nonlc_modelling/base/evaluation_summary.csv`;
`--model` selects the family, default `xgboost`, taking the validation row —
the fold set the cutoff was searched on). Pass `--cutoff` to override.

Optional: `--train-pred`, `--nonlc-subtypes` (joint `SampleID, nonlc_subtype`
table), `--dca-steps` (default 101), `--figures` (default: tables only).
Outputs:
`data/subgroup_metrics.csv`, `data/dca_{val,ext,train}.csv`, subgroup
box/bar figures and `figs/dca_curve.png`.

Subgroup dimensions whose source columns are missing from the merged data
(e.g. imaging columns when `--imaging-labels` only carries `SampleID`, or
`copd` when the clinical table has no COPD column) are skipped rather than
errors; the log reports how many dimensions were evaluated.

## `shap`

Refit the manuscript XGBoost model on the full discovery matrix and explain
it with `shap.TreeExplainer`.

```bash
python -m workflows.07_subgroup_analysis shap \
  --training-set /path/to/rfecv_selected_training_set.csv \
  --external-set /path/to/rfecv_selected_external_set.csv
```

Optional: `--model-parameters` (JSON), `--max-display` (default 20),
`--random-state`, `--figures` (default: tables only). Outputs:
`data/mean_abs_shap.csv`,
`figures/mean_abs_shap_bar.png`, `figures/shap_beeswarm_external.png`.

## `pr-calibration`

```bash
python -m workflows.07_subgroup_analysis pr-calibration \
  --val-pred /path/to/xgboost_val.csv --ext-pred /path/to/xgboost_ext.csv
```

Optional: `--n-bins` (10), `--n-bootstrap` (1000), `--random-state`
(default 42; propagated to PR, calibration, and Brier bootstrap intervals),
`--figures` (default: tables only).
Outputs: `data/pr_calibration_summary.csv`, `data/pr_curve_*.csv`,
`data/calibration_curve_*.csv`, `figures/pr_curves.png`,
`figures/calibration_curve_*.png`. Prediction tables accept integer labels or
the workflow 06 `"LC"`/`"nonLC"` strings (binarized internally).

## `npv-ppv-adjust`

```bash
python -m workflows.07_subgroup_analysis npv-ppv-adjust \
  --sensitivity 0.943 --specificity 0.874 \
  --n-sensitivity 345 --n-specificity 2540 \
  --prevalences 0.01 0.02 0.05 0.10
```

Optional: `--n-sensitivity`, `--n-specificity`, `--n-prevalence`,
`--method {bootstrap,wilson,bayesian}` (default bootstrap),
`--n-bootstrap` (10000), `--random-state`. Output:
`data/prevalence_adjusted_ppv_npv.csv` under the usual
`{output-dir}/07_subgroup_analysis/npv-ppv-adjust/`.

## `roc-compare`

```bash
python -m workflows.07_subgroup_analysis roc-compare \
  --pred /path/to/joined_predictions.csv \
  --model VOC=xgboost --model confounders=xgboost_proba
```

`--pred` carries `label` plus one probability column per `--model
NAME=COLUMN` (repeatable). `label` may be integer (1 = LC) or the workflow 06
`"LC"`/`"nonLC"` strings. Outputs: `data/roc_metrics.csv`,
`data/delong_tests.csv`, `data/roc_curve_*_<name>.csv`,
`figures/roc_comparison.png`.

## `confounder-models`

```bash
python -m workflows.07_subgroup_analysis confounder-models \
  --discovery-cinfo /path/to/cinfo_LC_nonLC_discovery.csv \
  --external-cinfo /path/to/cinfo_LC_nonLC_external.csv
```

Optional: `--eval-fold` (2), `--min-sensitivity` (0.95), `--cv` (5),
`--figures` (default: tables only). Outputs:
`data/{model}_*_pred_proba.csv`,
`data/confounder_metrics.csv`, `figures/confounder_roc.png`.

## `review-scenarios`

```bash
python -m workflows.07_subgroup_analysis review-scenarios \
  --lab-records /path/to/lab_records.csv \
  --ext-pred /path/to/xgboost_ext.csv
```

As in `subgroups`, `--cutoff` is optional and defaults to the model cutoff
recorded in the workflow 06 base `evaluation_summary.csv`.

Optional: `--sampling-sites`, `--imaging-labels` (enables the AJCC stage-I
scenario table). Outputs: `data/site_label_counts.csv` (when site labels
exist), `data/scenario_metrics.csv`, `data/ajcc_stage1_scenario_metrics.csv`
(with `--imaging-labels`). The lab-records contract is in
`docs/data_schema.md` (Lab Records Table). The collection-to-analysis
interval analysis was removed, so the lab-records table no longer needs
`sampling_time`/`analysis_time`.

## Source

Consolidated from the former workflows 07, 07.2 (removed), 08, 13, and
`cell_review_analysis`.
