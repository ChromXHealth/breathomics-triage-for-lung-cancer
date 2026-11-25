# Workflow 06: LC/non-LC Marker Model Training and Evaluation

## Purpose

Train and evaluate the classification model on the RFECV-selected feature
tables from workflow 05, at a sensitivity-anchored cutoff. Workflow 06 has
two modes, both pure modelling: feature identification (threshold
screening, manual qualitative identification and marker review, RFECV)
lives in workflow 05 (see `workflow_05.md`).

- **`base`**: the five-family model comparison — train logistic regression,
  random forest, kNN, XGBoost, and SVM on the discovery table with
  stratified k-fold cross-validation on identical fold partitions and
  produce per-family predictions, cutoffs, evaluation summaries, and ROC
  figures. The unadjusted
  XGBoost model is the one used downstream by workflow 07 (`subgroups`, `roc-compare`, `review-scenarios`).
- **`confounding`**: re-fit the same model family while adjusting for
  confounder covariates across a fixed set of adjustment schemes
  covariate adjustment schemes.

Both modes share the model registry and default hyperparameters
(`src/modelling.py`). The defaults record the end-point values of the
source notebooks' parameter search, so **no hyperparameter search runs at
runtime**; pass `--model-parameters` (JSON) to override any value.

## Mode `base`

### Purpose

Reproduce the source notebook's five-family comparison: train logistic
regression, random forest, kNN, XGBoost, and SVM on the RFECV-selected
discovery table with stratified k-fold cross-validation, using the searched
hyperparameters recorded in `src/modelling.py`. Every family sees identical
fold partitions (fixed CV splitter seed) and is evaluated at its own
sensitivity-anchored cutoff, with one internal/external ROC figure per
family. The unadjusted XGBoost model is the one used downstream by
workflow 07 (`subgroups`, `roc-compare`, `review-scenarios`).

### Inputs

- `--training-set`: workflow 05 `rfecv_selected_training_set.csv` (`SampleID, label, selected markers, age, gender`).
- `--external-set`: workflow 05 `rfecv_selected_external_set.csv`.
- `--postop-set`: workflow 05 `rfecv_selected_postOp_set.csv`.

### Execution

```bash
python -m workflows.06_lc_nonlc_modelling base \
  --training-set /path/to/differential_outputs/rfecv_selected_training_set.csv \
  --external-set /path/to/differential_outputs/rfecv_selected_external_set.csv \
  --postop-set /path/to/differential_outputs/rfecv_selected_postOp_set.csv
```

Useful optional arguments: `--models` (comma-separated subset of the five families; default: all five), `--cv` (default 5), `--eval-fold` (default 2), `--min-sensitivity` (default 0.95), `--n-bootstrap` (AUC intervals, default 1000), `--model-parameters` (JSON override applied to every family), `--random-state` (default 42; propagated to CV splitting, supported estimators, and bootstrap intervals), `--skip-figures`. An explicit `random_state` inside `--model-parameters` takes precedence for estimator fitting.

### Processing contract

- The modelling frame is `SampleID, label, selected markers`; covariate columns `age`/`gender` in the discovery and external tables are excluded from fitting, matching the source analysis.
- All families share identical fold partitions (CV splitter seed fixed at 42 unless `--random-state` is given), matching the source notebook's head-to-head comparison on identical folds. The source analysis evaluates a single held-out fold partition (`KFID == --eval-fold`; the source XGBoost analysis used fold 2), not fold-averaged scores; each family's cutoff is searched on its own validation partition with `min_sens=0.95` over 100 steps.
- Full fold-level prediction tables are exported alongside as `{model}_*_pred_proba.csv`.
- No prediction, label, or score is modified after model inference; the threshold is the only post-processing step.
- The exported prediction tables keep the original string labels (`LC`/`nonLC`); labels are binarized in memory for the cutoff search and metric computation only.

### Outputs (under `{output-dir}/06_lc_nonlc_modelling/base/`)

- `{model}_*_pred_proba.csv` (all folds) and `eval_batch_pred/{model}_*.csv` (evaluation fold); the XGBoost validation/external/postoperative partitions are the inputs to workflow 07 (`subgroups`, `roc-compare`, `review-scenarios`).
- `evaluation_summary.csv` — AUROC (with bootstrap intervals) and metrics per family and dataset, with each family's `sens>=0.95` cutoff as a dedicated column.
- `figs/{model}_roc.png` — internal/external ROC curves per family.

## Mode `confounding`

### Inputs

- `--training-set` / `--external-set`: workflow 05 `rfecv_selected_training_set.csv` / `rfecv_selected_external_set.csv`.
- `--train-pred` / `--val-pred`: base-mode `eval_batch_pred/{model}_train.csv` and `{model}_val.csv`; only their `SampleID` sets are used to rebuild the discovery train/validation partitions.
- `--discovery-cfd`: workflow 05 `cfd_dummy_mat_discovery.csv`.
- `--external-cfd`: workflow 05 `cfd_dummy_mat_external.csv`.

### Execution

```bash
python -m workflows.06_lc_nonlc_modelling confounding \
  --training-set /path/to/differential_outputs/rfecv_selected_training_set.csv \
  --external-set /path/to/differential_outputs/rfecv_selected_external_set.csv \
  --train-pred /path/to/modelling_outputs/06_lc_nonlc_modelling/base/eval_batch_pred/xgboost_train.csv \
  --val-pred /path/to/modelling_outputs/06_lc_nonlc_modelling/base/eval_batch_pred/xgboost_val.csv \
  --discovery-cfd /path/to/differential_outputs/cfd_dummy_mat_discovery.csv \
  --external-cfd /path/to/differential_outputs/cfd_dummy_mat_external.csv
```

Useful optional arguments: `--adjustments` (comma-separated subset of the default schemes; default: all), `--model` (default `xgboost`), `--min-sensitivity` (default 0.95), `--bootstrap-rounds` (default 1000), `--skip-figures`.

### Processing contract

- The discovery train/validation split is rebuilt from the base-mode evaluation partitions, so confounding-mode results are directly comparable to the base model.
- Each adjustment scheme fits the same model family once and searches the validation-set cutoff at `min_sens=0.95`; AUROC confidence intervals come from bootstrap resampling.
- Adjustment schemes and their covariate sets are defined in `src/confounding.py` (`DEFAULT_ADJUSTMENTS`).

### Outputs (under `{output-dir}/06_lc_nonlc_modelling/confounding/`)

- `data/{adjustment}_train_pred.csv`, `{adjustment}_val_pred.csv`, `{adjustment}_ext_pred.csv`
- `figs/{model}_{adjustment}_roc.png` (unless `--skip-figures`)
- `evaluation_summary.csv` — cutoff and metric intervals per adjustment and dataset

## Source

Derived from the original base-modelling and confounding-modelling
analyses. RFECV feature selection was split into workflow 05; the
interactive parameter-tuning cells are transcribed as explicit defaults in
`src/modelling.py`.
