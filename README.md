# Toward breathomics-guided pre-imaging triage for lung cancer detection

Analysis source code associated with the Cell manuscript *Toward
breathomics-guided pre-imaging triage for lung cancer detection*
(manuscript CELL-D-26-03180), including the figure code embedded in the
analysis workflows.

## Study Summary

Lung cancer remains the leading cause of cancer death, in large part because
most tumors are found too late. Screening with low-dose computed tomography
(LDCT) lowers mortality but strains resources and generates frequent false
positives. This study developed and validated a noninvasive breath test based
on 12 volatile organic compounds (VOCs) measured by thermal-desorption gas
chromatography–mass spectrometry (TD-GC-MS) in a multicenter study of
10,772 participants.

- Developed in a discovery cohort (n = 7,302), the locked model achieved an
  area under the curve (AUC) of 0.92 in the external cohort (n = 2,885).
- The model identified 96.4% of stage I and 94.6% of stage IA1 cancers with
  70.6% specificity.
- In a nested community subset, sensitivity was 72.7% and the negative
  predictive value reached 99.7%.
- Exploratory postoperative analysis showed lower model scores after
  resection.
- Aggregate proof-of-concept evaluation of portable micro-GC in an enriched
  cohort (n = 585) yielded an AUC of 0.90.

The study was registered at ClinicalTrials.gov as NCT06518655,
NCT06528418, and NCT06988943.

## Repository Scope

This repository contains the LC/non-LC analysis workflows for manuscript
steps 0–7 and their sub-steps, step 13, and the Cell review-response
analyses (including model explanation). Figures are produced inside the
analysis workflows from computed statistics.

The repository does not contain participant-level data, clinical labels,
trained model artifacts, prediction outputs, analysis results, or test code.
Numerical and graphical reproduction requires authorized access to the
corresponding study data.

All source code, configuration, documentation, and working comments in this
repository are written in English.

## Layout

- `src/`: reusable analysis modules.
- `workflows/`: ordered analysis entry points (00–08, 13, and the
  review-response workflows).
- `configs/`: data, model, evaluation, and figure configuration.
- `docs/`: data contracts, workflow documentation, and source mappings.

## Status

All planned workflows are migrated from the internal research notebooks
without changing model parameters, thresholds, sample definitions, or
reported results. Each workflow document in `docs/` lists any intentional
deviation from its source notebook.

## Requirements

Python 3.9.18 is the reference runtime. See `environment.yml` and
`requirements.lock` for the dependency specification. Create and activate an
isolated environment before running workflows.

## Execution

Run module entry points from this repository's root. This repository does
not resolve paths in any parent or sibling repository; every input and
output path must be supplied explicitly.

```bash
python -m workflows.00_formatting \
  --voc-input /path/to/voc_workbook.xlsx \
  --snr-input /path/to/snr_workbook.xlsx \
  --clinical-input /path/to/clinical_labels.csv \
  --output-dir /path/to/formatting_outputs
```

See `docs/workflow.md` and the numbered workflow documents for the full
command contract. The execution order is: formatting (00), SNR quality
control and sensitivity correction (01), batch correction (02), environmental
VOC analysis (03), data split (04), differential marker analysis (05),
modelling (06, 06.1), model and subgroup analysis (07, 07.2), risk-model
comparison (07.2), and the review-response analyses (13, 08,
`cell_review_analysis`).

## Data

No study data are distributed with this repository. Input contracts are
documented in `docs/data_schema.md`. Paths to authorized local inputs must
be supplied through command-line arguments.

## License

This repository is released under the MIT License; see `LICENSE`. The author
list for citation is maintained in `CITATION.cff`.

## Provenance

The mapping from each public module to its internal notebook source is
maintained in `docs/notebook_source_map.csv`. Figure-input principles are
documented in `docs/figure_inputs.md`.
