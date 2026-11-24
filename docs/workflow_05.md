# Workflow 05: LC/non-LC Differential Marker Analysis and Feature Selection

## Purpose

Perform the LC/non-LC discovery-set differential analysis and feature
selection. The screening pipeline has three stages:

1. **Threshold screening** — covariate encoding, per-VOC confounder-adjusted
   binomial GLMs, direction-adjusted AUCs, clinical covariate associations,
   and OPLS-DA feature importance flag candidate markers by the
   manuscript thresholds (FDR/VIP/AUROC).
2. **Qualitative identification review (manual)** — this workflow performs
   NO compound identification. Between threshold screening and RFECV, a
   human reviews every candidate: its annotation is confirmed against mass
   spectral libraries (per the manuscript Methods: NIST library matching
   with similarity and retention-index/elution-order constraints, plus a
   multi-spectrum consensus rule), and its metabolic and physiological
   plausibility is checked against the literature. Candidates that fail
   this review are excluded. The purpose is the manuscript's manual
   verification requirement: confirm that each candidate marker is
   consistent with metabolic logic, and remove markers inconsistent with
   physiological logic. The reviewed outcome is recorded in an authorized
   marker table and supplied via `--authorized-marker-table`.
3. **RFECV feature selection** — L1-logistic RFECV retains the final panel
   from the authorized candidates. Workflow 06 trains and evaluates the
   model on the RFECV-selected tables; it assumes stages 1–2 are complete.

## Inputs

All paths are caller-supplied:

- `--voc-input`: workflow 04 discovery VOC table.
- `--clinical-input`: workflow 04 discovery clinical table.
- `--external-voc`: workflow 04 external VOC table.
- `--external-clinical-input`: workflow 04 external clinical table.
- `--postoperative-voc`: workflow 01 `sens_corrected_postOp_voc.csv` table.
- `--precombat-voc`: sensitivity-corrected, pre-ComBat LC/non-LC measurement table.
- `--voc-info`: VOC annotation workbook containing `Alignment ID`.
- `--authorized-marker-table` (optional): the human-reviewed marker table
  produced by the qualitative identification step (stage 2). Contains a
  `VOC ID` column; a `marker_flag` column is honored if present. Without
  it, the flagged candidates are used unreviewed and an explicit note is
  logged — production runs must pass the reviewed table.
- `--output-dir`: destination directory.

The default clinical covariate contract contains `gender`, `age`, `BMI`, `dining_status`, `smoke_hist`, `second-hand_smoke_status`, `env_smoke_status`, `hazardous_substance_exposure`, `family_Ca_hist`, `lung_disease_hist`, `other_disease_hist`, and `Ca_hist`.

## Execution

Run from this repository's root:

```bash
python -m workflows.05_lc_nonlc_analysis \
  --voc-input /path/to/split_outputs/VOC_LC_nonLC_discovery.csv \
  --clinical-input /path/to/split_outputs/cinfo_LC_nonLC_discovery.csv \
  --external-voc /path/to/split_outputs/VOC_LC_nonLC_external.csv \
  --external-clinical-input /path/to/split_outputs/cinfo_LC_nonLC_external.csv \
  --postoperative-voc /path/to/qc_outputs/sens_corrected_postOp_voc.csv \
  --precombat-voc /path/to/qc_outputs/sens_corrected_voc_LC_nonLC_ENV.csv \
  --voc-info /path/to/voc_annotations.xlsx \
  --output-dir /path/to/differential_outputs
```

Useful optional arguments are `--authorized-marker-table` (reviewed marker
table from the manual identification step), `--positive-label`,
`--opls-search-range`, `--opls-cv`, `--cv` (RFECV folds), `--rfe-jobs`,
`--random-state`, `--fdr-threshold`, `--vip-threshold`, and
`--auroc-threshold`.

## Processing Contract

- Discovery age mean and standard deviation standardize discovery and external age fields.
- Exported covariate matrices fill missing categorical values with `UNK`; the OPLS matrix preserves source missing-value behavior.
- Each VOC is tested with `label ~ TARGET_VOC + clinical covariates`, with categorical covariates encoded in the formula.
- AUCs below 0.5 are direction-adjusted to the equivalent complemented score.
- Post-correction and pre-correction greater-than relations use one-sided Welch t-tests.
- Clinical covariates use chi-square tests, except age, which uses a two-sample t-test and direction-adjusted AUC.
- OPLS component search uses five-fold cross-validation by default and the manuscript search range of 50.
- VIP values are merged with univariate metrics, and marker selection applies FDR < 0.05, VIP > 1, and AUC > 0.6. `marker_metrics.csv` is the pre-review candidate list; it is not consumed downstream.
- Qualitative identification is manual and outside this repository (see Purpose). Its outcome enters through `--authorized-marker-table`; RFECV always runs on the authorized candidates.
- RFECV (L1 logistic, step 1, ROC-AUC scoring, stratified `--cv` folds) refines the authorized candidate markers on the discovery model frame; the external and postoperative frames are reduced to the same retained panel. The postoperative frame is log10(x + 1) transformed and z-scored, matching the source analysis.
- Discovery and external model frames carry `age` and `gender` covariate columns (z-scored age, male indicator); the postoperative frame does not.

## Outputs

- `cfd_dummy_mat_discovery.csv`
- `cfd_dummy_mat_external.csv`
- `cfd_dummy_mat.csv`
- `cfd_pval_auroc.csv`
- `voc_metrics.csv`
- `cfd_vip.csv`
- `marker_metrics.csv` — flagged candidate list for the manual qualitative identification review (not a downstream input)
- `rfecv_selected_training_set.csv`
- `rfecv_selected_external_set.csv`
- `rfecv_selected_postOp_set.csv`
- `rfecv_selected_markers.csv` — the final RFECV-retained panel (for reference and audit)

The OPLS implementation is vendored in `src/opls.py` and depends on the public `pyopls` package; no private package import remains.

## Exclusions

Manual curated-marker spreadsheets, Chinese display labels, plotting-only cells, and LC/HC or LC/LN branches are not part of this workflow.
