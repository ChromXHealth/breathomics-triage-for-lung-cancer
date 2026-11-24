# Workflow 00: Upstream Data Preparation

## Purpose

Define the data assets a caller must prepare before running the analysis
workflows. This workflow has no executable entry point in this repository:
format conversion, clinical eligibility filtering, and dataset separation
happen outside these workflows, because instrument exports and eligibility
rules differ between studies. This document explains what to prepare; the
column-level contracts live in `data_schema.md`.

## Preparation Checklist

Prepare five CSV tables before running workflow 01. Column names are
case-sensitive; every table uses `SampleID` as the sample identifier.

1. **LC/non-LC VOC measurement table** — one row per breath sample:
   `SampleID`, `label` (`LC` or `nonLC`), then the VOC feature columns.
   Apply clinical eligibility filtering first: records without a valid LC/non-LC
   label do not belong in this table.
2. **LC/non-LC SNR table** — the SNR twin of table 1: same samples, same row
   order, same feature columns, with SNR values instead of abundances.
3. **Environmental VOC table** — one row per environmental measurement:
   `SampleID` plus the VOC feature columns. No `label`; workflow 01 assigns
   `ENV` and aligns the features to the QC-retained columns.
4. **Postoperative follow-up VOC table** — one row per follow-up sample:
   `SampleID` plus the VOC feature columns. No `label`; workflow 01 assigns
   `postOp` and aligns the retained features.
5. **Clinical label table** — one row per participant: `SampleID`,
   `LC_nonLC` (`LC` or `nonLC`), `label_postOp` (`T`/`F`), `smoke_hist`,
   and `BMI` (numeric or `UNK`); `height` and `weight` are optional and
   dropped by workflow 04. Upstream eligibility filtering is assumed
   complete.

## Additional Assets for Workflow 03

Two more inputs are needed by workflow 03 (environmental VOC analysis):

6. **Breath and environmental sampling-location tables** — one row per
   sample: `SampleID` and `SampleLoc` (header names overridable through
   workflow 03's column arguments). Rows with missing or `UNK` locations are
   dropped; at least one site must hold both breath and environmental
   samples with more than three records per group.
7. **VOC annotation workbook** (xlsx) — the instrument software's
   annotation export: `Alignment ID` (unique join key), `Average Rt(min)`,
   `Quant mass`, and `S/N average`; additional annotation columns are kept
   in the outputs.

## Rules

- Tables 1 and 2 must contain the same samples in the same row order;
  sample-level QC results join them positionally.
- Environmental and follow-up tables carry no labels; label assignment is
  the workflows' job, not the caller's.
- The LC/non-LC label definition (`LC` versus `nonLC`) is part of the study
  analysis contract: callers reproduce it during preparation, and downstream
  workflows never re-derive or modify it.
- Detailed column types and constraints: `data_schema.md`.

## Execution

No command. This workflow is a contract; the first executable step is
workflow 01, whose `--voc-input`, `--snr-input`, `--environmental-voc`, and
`--followup-voc` arguments consume tables 1–4, and workflow 04 consumes
table 5. Workflow 03 consumes the sampling-location tables and the VOC
annotation workbook from the additional-assets section.

## Migration Notes

This document replaces the executable workflow 00 (MS-DIAL formatting) from
the original migration. The transpose-based formatting code was removed
because readers' instrument exports need not follow the MS-DIAL layout, and
the study-specific preparation steps (eligibility filtering,
environmental/follow-up separation, LC/non-LC label derivation) are the
caller's responsibility. The removed code is recorded as excluded in the migration notes.
