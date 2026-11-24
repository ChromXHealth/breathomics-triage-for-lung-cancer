# Workflow 01: LC/non-LC SNR Quality Control and Sensitivity Correction

## Purpose

Apply feature- and sample-level SNR quality control, construct environmental inputs, and apply device- and acquisition-time-specific sensitivity factors for LC/non-LC and follow-up analysis.

## Inputs

All paths are caller-supplied. Upstream data preparation (format conversion, clinical eligibility filtering, and the breath/environmental separation) is assumed complete; the input contracts are defined in `data_schema.md`.

- `--voc-input`: LC/non-LC VOC measurement table (`SampleID`, `label`, VOC feature columns; include the configured extra feature if one is set).
- `--snr-input`: LC/non-LC SNR table with the same layout and row order as `--voc-input`.
- `--environmental-voc`: environmental VOC measurement table (`SampleID`, VOC feature columns).
- `--followup-voc`: postoperative follow-up VOC measurement table (`SampleID`, VOC feature columns).
- `--analysis-time`: acquisition metadata table providing `SampleID`, `DeviceID`, `AnalysisDate`, and `AnalysisTime`.
- `--sensitivity-factors`: device/time table with `DeviceID`, `AnalysisTime`, `NextTime`, and `sensitivity_factor`.
- `--preprocessing-config`: YAML file with the `snr_qc` parameter block (SNR threshold, call-rate thresholds, optional extra feature). Defaults to `configs/preprocessing.yaml`; study parameters are versioned there without recalibration.
- `--output-dir`: destination directory.

## Execution

Run from this repository's root:

```bash
python -m workflows.01_snr_qc \
  --voc-input /path/to/raw_voc_LC_nonLC.csv \
  --snr-input /path/to/raw_snr_LC_nonLC.csv \
  --environmental-voc /path/to/ENV-VOC-RAW.csv \
  --followup-voc /path/to/raw_voc_followup.csv \
  --analysis-time /path/to/acquisition_metadata.csv \
  --sensitivity-factors /path/to/sensitivity_factors.csv \
  --output-dir /path/to/qc_outputs
```

Optional sample exclusions can be supplied by repeating `--exclude-sample SAMPLE_ID`.

## Quality-Control Rules

All thresholds and the optional extra feature come from the `snr_qc` block of `--preprocessing-config`.

- An SNR call is positive when SNR is strictly greater than the configured threshold.
- A VOC is retained when its call rate meets the configured threshold in any label-specific column or the `ALL` column.
- The configured extra feature (if any) is retained explicitly regardless of call rate and is excluded from the sample call-rate denominator.
- A sample is retained when its call rate across QC-selected features meets the configured threshold.
- LC/non-LC measurements remain unimputed after SNR filtering.
- Environmental inputs use the legacy detected-value-minimum imputation helper.

Independent LC/HC and LC/LN quality-control branches are intentionally excluded.
