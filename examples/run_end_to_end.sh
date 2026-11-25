#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA="${SYNTHETIC_DATA_DIR:-$ROOT/examples/synthetic-data}"
OUT="${SYNTHETIC_OUTPUT_DIR:-$ROOT/examples/synthetic-output}"
PYTHON="${PYTHON:-python}"

"$PYTHON" examples/generate_synthetic_data.py --output-dir "$DATA" --seed 42

"$PYTHON" -m workflows.01_snr_qc \
  --voc-input "$DATA/raw_voc_LC_nonLC.csv" \
  --snr-input "$DATA/raw_snr_LC_nonLC.csv" \
  --environmental-voc "$DATA/ENV-VOC-RAW.csv" \
  --followup-voc "$DATA/raw_voc_followup.csv" \
  --analysis-time "$DATA/acquisition_metadata.csv" \
  --sensitivity-factors "$DATA/sensitivity_factors.csv" \
  --output-dir "$OUT"

"$PYTHON" -m workflows.02_batch_analysis \
  --input "$OUT/01_snr_qc/sens_corrected_voc_LC_nonLC_ENV.csv" \
  --batch-config "$DATA/batches.yaml" \
  --output-dir "$OUT"

"$PYTHON" -m workflows.03_environmental_voc \
  --corrected-voc "$OUT/02_batch_analysis/combat_voc_LC_nonLC_ENV.csv" \
  --breath-locations "$DATA/breath_locations.csv" \
  --environment-locations "$DATA/environment_locations.csv" \
  --voc-info "$DATA/voc_annotations.xlsx" \
  --output-dir "$OUT"

"$PYTHON" -m workflows.04_data_split \
  --voc-input "$OUT/03_environmental_voc/evoc_free_LC_nonLC.csv" \
  --clinical-input "$DATA/clinical_labels.csv" \
  --split-config "$DATA/cohort_split.yaml" \
  --output-dir "$OUT"

"$PYTHON" -m workflows.05_lc_nonlc_analysis \
  --voc-input "$OUT/04_data_split/VOC_LC_nonLC_discovery.csv" \
  --clinical-input "$OUT/04_data_split/cinfo_LC_nonLC_discovery.csv" \
  --external-voc "$OUT/04_data_split/VOC_LC_nonLC_external.csv" \
  --external-clinical-input "$OUT/04_data_split/cinfo_LC_nonLC_external.csv" \
  --postoperative-voc "$OUT/01_snr_qc/sens_corrected_postOp_voc.csv" \
  --precombat-voc "$OUT/01_snr_qc/sens_corrected_voc_LC_nonLC_ENV.csv" \
  --voc-info "$DATA/voc_annotations.xlsx" \
  --authorized-marker-table "$DATA/authorized_markers.csv" \
  --opls-search-range 2 --opls-cv 2 --cv 2 --rfe-jobs 1 \
  --random-state 42 --output-dir "$OUT"

"$PYTHON" -m workflows.06_lc_nonlc_modelling base \
  --training-set "$OUT/05_lc_nonlc_analysis/rfecv_selected_training_set.csv" \
  --external-set "$OUT/05_lc_nonlc_analysis/rfecv_selected_external_set.csv" \
  --postop-set "$OUT/05_lc_nonlc_analysis/rfecv_selected_postOp_set.csv" \
  --cv 2 --eval-fold 1 --n-bootstrap 20 --random-state 42 \
  --skip-figures --output-dir "$OUT"

"$PYTHON" -m workflows.06_lc_nonlc_modelling confounding \
  --training-set "$OUT/05_lc_nonlc_analysis/rfecv_selected_training_set.csv" \
  --external-set "$OUT/05_lc_nonlc_analysis/rfecv_selected_external_set.csv" \
  --train-pred "$OUT/06_lc_nonlc_modelling/base/eval_batch_pred/xgboost_train.csv" \
  --val-pred "$OUT/06_lc_nonlc_modelling/base/eval_batch_pred/xgboost_val.csv" \
  --discovery-cfd "$OUT/05_lc_nonlc_analysis/cfd_dummy_mat_discovery.csv" \
  --external-cfd "$OUT/05_lc_nonlc_analysis/cfd_dummy_mat_external.csv" \
  --adjustments ALL --bootstrap-rounds 20 --random-state 42 \
  --skip-figures --output-dir "$OUT"

"$PYTHON" -m workflows.07_subgroup_analysis subgroups \
  --val-pred "$OUT/06_lc_nonlc_modelling/base/eval_batch_pred/xgboost_val.csv" \
  --ext-pred "$OUT/06_lc_nonlc_modelling/base/eval_batch_pred/xgboost_ext.csv" \
  --imaging-labels "$DATA/imaging_labels.csv" \
  --clinical-discovery "$OUT/04_data_split/cinfo_LC_nonLC_discovery.csv" \
  --clinical-external "$OUT/04_data_split/cinfo_LC_nonLC_external.csv" \
  --nonlc-subtypes "$DATA/nonlc_subtypes.csv" \
  --evaluation-summary "$OUT/06_lc_nonlc_modelling/base/evaluation_summary.csv" \
  --dca-steps 20 --output-dir "$OUT"

"$PYTHON" -m workflows.07_subgroup_analysis shap \
  --training-set "$OUT/05_lc_nonlc_analysis/rfecv_selected_training_set.csv" \
  --external-set "$OUT/05_lc_nonlc_analysis/rfecv_selected_external_set.csv" \
  --random-state 42 --output-dir "$OUT"

"$PYTHON" -m workflows.07_subgroup_analysis pr-calibration \
  --val-pred "$OUT/06_lc_nonlc_modelling/base/eval_batch_pred/xgboost_val.csv" \
  --ext-pred "$OUT/06_lc_nonlc_modelling/base/eval_batch_pred/xgboost_ext.csv" \
  --n-bootstrap 20 --random-state 42 --output-dir "$OUT"

"$PYTHON" -m workflows.07_subgroup_analysis npv-ppv-adjust \
  --sensitivity 0.90 --specificity 0.80 \
  --n-sensitivity 40 --n-specificity 40 --n-prevalence 120 \
  --prevalences 0.01 0.05 0.10 --n-bootstrap 100 \
  --random-state 42 --output-dir "$OUT"

"$PYTHON" - "$OUT" <<'PY'
from pathlib import Path
import sys
import pandas as pd

base = Path(sys.argv[1]) / "06_lc_nonlc_modelling/base/eval_batch_pred"
left = pd.read_csv(base / "xgboost_ext.csv")
right = pd.read_csv(base / "logistic_ext.csv")
joined = left[["SampleID", "label", "pred"]].rename(columns={"pred": "xgboost"})
joined["logistic"] = right["pred"]
joined.to_csv(base.parent.parent.parent / "07_subgroup_analysis/joined_predictions.csv", index=False)
PY

"$PYTHON" -m workflows.07_subgroup_analysis roc-compare \
  --pred "$OUT/07_subgroup_analysis/joined_predictions.csv" \
  --model VOC=xgboost --model Logistic=logistic \
  --n-bootstrap 20 --random-state 42 --output-dir "$OUT"

"$PYTHON" -m workflows.07_subgroup_analysis confounder-models \
  --discovery-cinfo "$OUT/04_data_split/cinfo_LC_nonLC_discovery.csv" \
  --external-cinfo "$OUT/04_data_split/cinfo_LC_nonLC_external.csv" \
  --cv 2 --eval-fold 1 --random-state 42 --output-dir "$OUT"

"$PYTHON" -m workflows.07_subgroup_analysis review-scenarios \
  --lab-records "$DATA/lab_records.csv" \
  --ext-pred "$OUT/06_lc_nonlc_modelling/base/eval_batch_pred/xgboost_ext.csv" \
  --evaluation-summary "$OUT/06_lc_nonlc_modelling/base/evaluation_summary.csv" \
  --imaging-labels "$DATA/imaging_labels.csv" --output-dir "$OUT"

"$PYTHON" - "$OUT" <<'PY'
from pathlib import Path
import sys

out = Path(sys.argv[1])
expected = [
    "01_snr_qc/sens_corrected_voc_LC_nonLC_ENV.csv",
    "02_batch_analysis/combat_voc_LC_nonLC_ENV.csv",
    "03_environmental_voc/evoc_free_LC_nonLC.csv",
    "04_data_split/VOC_LC_nonLC_discovery.csv",
    "05_lc_nonlc_analysis/rfecv_selected_training_set.csv",
    "06_lc_nonlc_modelling/base/evaluation_summary.csv",
    "06_lc_nonlc_modelling/confounding/evaluation_summary.csv",
    "07_subgroup_analysis/subgroups/data/subgroup_metrics.csv",
    "07_subgroup_analysis/shap/data/mean_abs_shap.csv",
    "07_subgroup_analysis/pr-calibration/data/pr_calibration_summary.csv",
    "07_subgroup_analysis/npv-ppv-adjust/data/prevalence_adjusted_ppv_npv.csv",
    "07_subgroup_analysis/roc-compare/data/roc_metrics.csv",
    "07_subgroup_analysis/confounder-models/data/confounder_metrics.csv",
    "07_subgroup_analysis/review-scenarios/data/scenario_metrics.csv",
]
missing = [name for name in expected if not (out / name).is_file()]
if missing:
    raise SystemExit(f"Missing smoke outputs: {missing}")
print(f"Smoke run passed: {len(expected)} key outputs verified under {out}")
PY
