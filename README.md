# Toward breathomics-guided pre-imaging triage for lung cancer detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24504321.svg)](https://doi.org/10.5281/zenodo.24504321)

Analysis source code associated with the Cell manuscript *Toward
breathomics-guided pre-imaging triage for lung cancer detection*, including
the figure code embedded in the analysis workflows.

> **Scope of this release.** This repository fully provides the code for
> data processing (workflow 01–03), cohort splitting (04), marker discovery
> (05), and model training and evaluation (06). The subgroup and post-hoc
> analyses (workflow 07) and some figure generation depend on sensitive
> clinical and imaging information (imaging/pathology labels, sampling-site
> records) that cannot be shared, so those parts are provided as reference
> code only and are not fully reproducible without the study data.

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

This repository contains the LC/non-LC analysis workflows: data preparation
and quality control (01–04), marker analysis and modelling (05–06), and the
subgroup / post-hoc analyses (07). Figures are produced inside the analysis
workflows from computed statistics.

The repository does not contain participant-level data, clinical labels,
trained model artifacts, prediction outputs, analysis results, or test code.
Numerical and graphical reproduction requires authorized access to the
corresponding study data.

All source code, configuration, documentation, and working comments in this
repository are written in English.

## Layout

- `src/`: reusable analysis modules.
- `workflows/`: ordered analysis entry points (01–07); workflow 07 is a
  subcommand-style collection of subgroup and post-hoc analyses.
- `configs/`: data, model, evaluation, and figure configuration.
- `docs/`: data contracts, workflow documentation, and source mappings
  (start with `docs/README.md` for the documentation guide).

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

Upstream data preparation (instrument formatting, clinical eligibility
filtering, and the breath/environmental separation) is assumed complete
before the workflows run; the preparation contract is documented in
`docs/workflow_00.md` (contract only, no executable entry point) and the
column-level contracts in `docs/data_schema.md`.

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

See `docs/README.md` for the documentation guide, then the numbered
workflow documents for the full command contract. The execution order is:
SNR quality control and sensitivity correction (01), batch correction (02),
environmental VOC analysis (03), data split (04), differential marker
analysis (05), modelling (06), and subgroup / post-hoc analyses (07).

## Data

No study data are distributed with this repository. Input contracts are
documented in `docs/data_schema.md`. Paths to authorized local inputs must
be supplied through command-line arguments.

## License

This repository is released under the MIT License; see `LICENSE`. The author
list for citation is maintained in `CITATION.cff`.

## Figure Inputs

Figure-input principles are documented in `docs/figure_inputs.md`.
