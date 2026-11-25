# Pre-imaging stratification for lung cancer detection based on breath exhalate

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24504320.svg)](https://doi.org/10.5281/zenodo.24504320)

Analysis source code associated with the Cell manuscript *Pre-imaging
stratification for lung cancer detection based on breath exhalate*, including
the figure code embedded in the analysis workflows.

This repository is shared for research transparency and as a reference for
other researchers studying the analysis. It is not intended to be a turnkey or
general-purpose software package. External contributions are not currently
accepted.

> **Scope of this release.** Workflows 01–06 are provided in full as
> executable analysis code. Workflow 07 and some figure generation are also
> provided, but depend on sensitive clinical and imaging information. Public
> synthetic-data tests verify execution, not the manuscript's numerical
> results. Numerical and graphical reproduction requires authorized access to
> the study data and the study-specific assets identified in the documentation.

## Study Summary

Lung cancer remains the leading cause of cancer death, in large part because
most tumors are found too late. Screening with low-dose computed tomography
(LDCT) lowers mortality but strains resources and generates frequent false
positives. This study developed and validated a noninvasive breath test based
on 12 volatile organic compounds (VOCs) measured by thermal-desorption gas
chromatography–mass spectrometry (TD-GC-MS) in a multicenter study of
10,772 participants.

- Developed in a discovery cohort (n = 7,302), the locked model achieved an
  area under the curve (AUC) of 0.92 in the mixed external cohort
  (n = 2,885).
- At a cutoff of 0.27, the model identified 96.4% of stage I and 94.6% of
  stage IA1 cancers with 70.6% specificity.
- In the nested community-screening subset at a cutoff of 0.75, sensitivity
  was 72.7% and the negative predictive value was 99.7%.
- Exploratory post-operative analysis showed lower model scores after tumour
  resection.
- Aggregate proof-of-concept evaluation of portable micro-GC in an enriched
  cohort (n = 585) yielded an AUC of 0.90.

The study was registered at ClinicalTrials.gov as NCT06518655,
NCT06528418, and NCT06988943.

If any study description or numerical value in this repository differs from
the associated manuscript, the manuscript is authoritative.

## Repository Scope

This repository contains the LC/non-LC analysis workflows: data preparation
and quality control (01–04), marker analysis and modelling (05–06), and the
subgroup / post-hoc analyses (07). Figures are produced inside the analysis
workflows from computed statistics.

This GitHub repository does not itself contain participant-level data tables,
clinical labels, trained model artifacts, prediction outputs, or analysis
results. A separate public raw-data example is described under “Data
Availability.” Study-specific cohort assignments and the final reviewed marker
table must be supplied by an authorized user. Public templates document their
schemas; they do not encode the source study's confidential assets.

All source code, configuration, workflow documentation, and working comments
in this repository are written in English.

## Layout

- `src/`: reusable analysis modules.
- `workflows/`: ordered analysis entry points (01–07); workflow 07 is a
  subcommand-style collection of subgroup and post-hoc analyses.
- `configs/`: preprocessing and batch parameters plus schemas for
  study-specific cohort and marker assets.
- `docs/`: data contracts, workflow documentation, and figure-input rules
  (start with `docs/README.md` for the documentation guide).
- `examples/`: synthetic-data generation and an end-to-end smoke runner.
- `tests/`: unit, contract, reproducibility, and smoke tests.

## Analysis Status

The workflows were migrated from the internal research notebooks without
intentionally retuning model parameters, thresholds, or sample definitions.
Known differences are documented in each workflow. In particular, workflow
02 fits ComBat on the public LC/non-LC + environmental correction basis
instead of the original joint-cohort basis; numerical equivalence with the
original correction is therefore not asserted.

## Requirements and Tested Platform

The reference runtime is Python 3.9.18. This repository was prepared and
tested on **macOS 27.2.1 (26B131)** using the `breath-voc-analysis` Conda
environment. Other operating systems and platform/version combinations have
not been tested. This describes the current verification scope and does not
imply that other platforms are incompatible.

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate breath-voc-analysis
python -m pip check
```

If the environment already exists, only the activation command is needed.
`requirements.txt` records the complete direct and transitive runtime
dependency closure resolved from the reference environment. The Conda file
pins the direct analysis dependencies; both describe the release runtime.
Test-only tools are intentionally excluded from both release dependency files.

## Optional Synthetic Execution Check

The optional synthetic example contains no participant-derived data. It checks
input preparation, workflow interfaces, output contracts, and model execution
with reduced computation settings; it is not a supported application setup.

```bash
conda activate breath-voc-analysis
python examples/generate_synthetic_data.py --output-dir examples/synthetic-data
bash examples/run_end_to_end.sh
```

Generated inputs and outputs are ignored by Git. To choose another location,
set `SYNTHETIC_DATA_DIR` and `SYNTHETIC_OUTPUT_DIR` when calling the runner;
see `examples/README.md`.

For development only, install the test tools and run the automated tests:

```bash
python -m pip install pytest==8.4.2 ruff==0.17.6
python -m pytest -q
python -m ruff check src workflows tests examples
```

## Execution with Authorized Study Data

Run module entry points from this repository's root. Every input and output
path must be supplied explicitly; no parent or sibling repository is searched.

Upstream data preparation (instrument formatting, clinical eligibility
filtering, and the breath/environmental separation) is assumed complete
before the workflows run. The preparation contract is documented in
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

See `docs/README.md` for the documentation guide, then the numbered workflow
documents for the full command contract. The execution order is SNR quality
control and sensitivity correction (01), batch correction (02), environmental
VOC analysis (03), data split (04), differential marker analysis (05),
modelling (06), and subgroup / post-hoc analyses (07).

## Data Availability

### Public example dataset

A public example comprising **200 exhaled-breath samples from the External
Test Set** is available from Mendeley Data:

- Dataset: <https://data.mendeley.com/datasets/g7w5yjsr9g/1>
- DOI: <https://doi.org/10.17632/g7w5yjsr9g.1>

The archive contains raw Agilent GC-MS acquisition files (`.d` format) only.
It does not include clinical labels, demographic variables, or derived feature
matrices. The raw files require upstream instrument formatting before they can
be used with the analysis contracts in `docs/data_schema.md`; by themselves,
they do not enable end-to-end reproduction of the manuscript analysis.

### Controlled additional resources

For participant-privacy and commercial-confidentiality reasons, the complete
dataset, participant labels and clinical records, trained models, and related
study-specific assets are **not publicly distributed by default**. Researchers
may contact the authors to request access. Provision is subject to review and
approval through the process described in the associated manuscript's Data
Availability statement; submitting a request does not guarantee access. Do not
attach participant-level or otherwise sensitive records to GitHub issues,
pull requests, or other public channels.

## Reproducibility Levels

- **Execution reproducibility:** public synthetic fixtures, deterministic
  random seeds, unit tests, and smoke workflows verify the released code.
- **Numerical reproduction:** authorized study data, the source-study cohort
  split, and the final reviewed marker table are required. Documented public
  workflow deviations must be considered when comparing results.

## Citation and Versioning

The author list and software citation are maintained in `CITATION.cff`. The
README badge uses the Zenodo **concept DOI**, which resolves to the latest
archived version. Each GitHub release receives its own version-specific DOI.
Please cite both the associated manuscript and the software version used.

The current development version is **v1.0.2-rc**; changes relative to the
initial archive are listed in `CHANGELOG.md`.

## License

This repository is released under the MIT License; see `LICENSE`.

## Figure Inputs

Figure-input principles are documented in `docs/figure_inputs.md`.
