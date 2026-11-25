# Documentation Guide

This directory holds the documents needed to understand and execute the
analysis workflows: workflow contracts, data contracts, and figure-generation
rules. Public synthetic examples require no study data. A separate public
Mendeley Data archive provides raw GC-MS files for 200 External Test Set
samples ([DOI: 10.17632/g7w5yjsr9g.1](https://doi.org/10.17632/g7w5yjsr9g.1)),
without labels or derived matrices. Numerical reproduction requires authorized
full study data and study-specific assets.

> **Scope of this release.** Workflows 01–06 are provided in full as
> executable analysis code. Workflow 07 and some figure generation depend on
> sensitive clinical/imaging information. Public synthetic tests verify
> execution and interfaces only. Numerical reproduction requires authorized
> study data and the study-specific cohort and reviewed-marker assets.

## Reading Order

### First pass: the big picture

1. `../README.md` — study summary, research-reference scope, tested platform,
   installation, optional synthetic execution check, and execution conventions.
2. [The workflow at a glance](#the-workflow-at-a-glance) — execution order,
   migration principles, and verification status.
3. `data_schema.md` — data contracts for the measurement, clinical, imaging,
   and prediction tables. Read before preparing any input data.
4. `../examples/README.md` — a fully public execution example.

### Second pass: every document, in execution order

Read the `workflow_*.md` files in execution order and consult each entry point
in `workflows/` plus its configuration in `configs/`:

| Document | Entry point | Purpose |
|---|---|---|
| `workflow_00.md` | — (preparation contract) | Data assets to prepare before running the workflows |
| `data_schema.md` | — | Column-level input contracts |
| `workflow_01.md` | `workflows/01_snr_qc.py` | Feature- and sample-level SNR QC and sensitivity correction |
| `workflow_02.md` | `workflows/02_batch_analysis.py` | Normalization and optional ComBat batch correction |
| `workflow_03.md` | `workflows/03_environmental_voc.py` | Environmental VOC identification and removal |
| `workflow_04.md` | `workflows/04_data_split.py` | Eligibility filtering and discovery/external split construction |
| `workflow_05.md` | `workflows/05_lc_nonlc_analysis.py` | Differential analysis, manual marker review interface, and RFECV |
| `workflow_06.md` | `workflows/06_lc_nonlc_modelling.py` | Five-family comparison and confounder-adjustment modelling |
| `workflow_07.md` | `workflows/07_subgroup_analysis.py` | Subgroup and post-hoc analysis subcommands |
| `figure_inputs.md` | — | Figure-generation contracts |

Executable workflow documents describe **Purpose**, **Inputs**, **Execution**,
processing rules, and known exclusions or deviations. `workflow_00.md` and
`data_schema.md` are contract-only documents with no entry point.

## The Workflow at a Glance

### Execution Order

Upstream preparation is assumed complete before step 1:

1. Apply signal-to-noise quality control and sensitivity correction.
2. Assess and correct batch effects.
3. Analyze environmental VOC measurements.
4. Create and record analysis splits.
5. Perform differential marker analysis and reviewed-panel feature selection.
6. Train and evaluate classification models.
7. Perform confounding-adjusted modelling and subgroup/post-hoc analyses.

### Migration Principles

- Preserve manuscript parameters, thresholds, sample definitions, and
  statistical methods unless an intentional deviation is documented.
- Transcribe implicit notebook parameters into versioned configuration.
- Do not retune models or modify reported results as part of code migration.
- Remove private package dependencies by migrating implementations into `src/`.
- Use immutable source fields for subgroup membership and cohort assignment.
- Exclude ad hoc record-modification pipelines and downstream dependencies.
- Keep confidential cohort assignments and other controlled assets out of Git;
  provide public schemas and synthetic fixtures instead.

### Verification Status

Workflows 01–07 are migrated, compile-checked, and CLI-verified in the
reference environment. Public synthetic-data smoke tests exercise every
workflow end to end, and automated tests cover key contracts and deterministic
behavior. These tests do not reproduce manuscript numerical results. Known
intentional deviations are documented in each `docs/workflow_*.md` file.

## Conventions

- All input and output paths are caller-supplied. `--output-dir` is a **base**
  directory: each workflow writes into a fixed subdirectory under it. The
  default base is `./analysis-output`; directories are created when missing.
- Every workflow prints structured run logs and identifies final outputs for
  downstream use.
- Study-specific analysis parameters are versioned in `configs/`.
  `preprocessing.yaml` and `batches.yaml` contain source-study preprocessing
  parameters. `cohort_split.example.yaml` and
  `authorized_markers.example.csv` document schemas only; they are not the
  source study's confidential assets and are not loaded by default.
- ComBat correction is optional and runs only when a batch config is passed to
  workflow 02. The cohort split is used only when explicitly supplied.
- Qualitative VOC identification is a manual step between threshold screening
  and RFECV. Its authorized outcome enters workflow 05 through
  `--authorized-marker-table`; production reproduction requires the final
  reviewed table.
- Column names are case-sensitive. Missing or duplicated required fields must
  produce an explicit error. Identifiers used for one-to-one joins must be
  unique.
- Randomized workflows expose `--random-state`, defaulting to 42. Bootstrap
  and model-fitting seeds are propagated through the public workflows.
- Every intentional deviation from a source notebook is documented in the
  corresponding workflow document.
