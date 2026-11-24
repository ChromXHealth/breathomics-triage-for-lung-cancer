# Workflow 02: LC/non-LC Batch Correction

## Purpose

Reproduce the data normalization and ComBat batch correction required to
generate the manuscript's LC/non-LC corrected matrix.

## Intentional Deviation from the Source Notebook

The source correction was fitted on a joint dataset constructed from three
study cohorts (LC/non-LC, LC/HC, LC/LN), with the LC/HC and LC/LN matrices
used only as upstream inputs to preserve that correction. This repository
supports the **LC/non-LC cohort only** (project decision), so ComBat is
fitted on the LC/non-LC + environmental matrix alone. The batch definitions
(device- and acquisition-time rules) are unchanged, but the correction basis
is smaller than in the source analysis. Cross-cohort label harmonization is
not applicable and was removed along with the LC/HC and LC/LN inputs.

## Inputs

- `--input`: workflow 01 `sens_corrected_voc_LC_nonLC_ENV.csv` table.
- `--batch-config` (optional): YAML file with the batch assignment rules.
  When omitted, ComBat correction is skipped and the normalized matrix is
  exported instead; downstream workflows can consume it the same way.
  Pass `configs/batches.yaml` to reproduce the source study's batch
  definitions.
- `--output-dir`: caller-selected base output directory; results are written
  under `{output-dir}/02_batch_analysis/` (default `./analysis-output`).

## Batch Labels

ComBat requires a batch label per sample and therefore a batch config; the
normalization and export steps do not. When `--batch-config` is supplied,
batches are assigned from device- and acquisition-time rules (per-device
ordered rules with a `"*"` fallback; see `configs/batches.yaml` for the
schema). The committed rules reproduce the source study's devices and
acquisition-time splits only — they are **study-specific**. Users analyzing
their own data must determine their acquisition batches separately (for
example from instrument run logs, QC drift, or known run boundaries) and edit
the config accordingly; the workflow refuses samples from devices with no
matching rule and no `'*'` fallback.

## Execution

Run from this repository's root:

```bash
# Normalize only (no batch config required):
python workflows/02_batch_analysis.py \
  --input analysis-output/01_snr_qc/sens_corrected_voc_LC_nonLC_ENV.csv

# Reproduce the source study's ComBat correction:
python workflows/02_batch_analysis.py \
  --input analysis-output/01_snr_qc/sens_corrected_voc_LC_nonLC_ENV.csv \
  --batch-config configs/batches.yaml
```

## Processing Contract

- Features are sorted by numeric feature identifier.
- Measurements undergo log10(x + 1) transformation and feature-wise standardization.
- With a batch config, batch labels are assigned from its device- and
  acquisition-time rules (see "Batch Labels" above); the committed values
  reproduce the source study's batch definitions.
- Without a batch config, ComBat is skipped and the downstream export keeps
  the normalized values.
- The full matrix is exported with batch metadata (ComBat path only), followed
  by the LC/non-LC export whose labels other than LC and ENV are mapped to
  nonLC.
