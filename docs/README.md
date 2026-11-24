# Documentation Guide

This directory holds the documents needed to understand and execute the
analysis workflows in this repository: the workflow contracts, the data
contracts, and the figure-generation rules. No document assumes access to
the study data.

> **Scope of this release.** Workflows 01–06 (data processing, cohort
> splitting, marker discovery, modelling) are fully provided and
> reproducible. Workflow 07 (subgroup and post-hoc analyses) and some
> figure generation rely on sensitive clinical/imaging information that
> cannot be shared; they are documented and provided as reference code,
> but not fully reproducible without the study data.

## Reading Order

### First pass: the big picture

1. `../README.md` (repository root) — study summary, repository scope,
   directory layout, and the execution conventions.
2. [The workflow at a glance](#the-workflow-at-a-glance) (this file, below)
   — the execution order, the migration principles, and the current
   execution status.
3. `data_schema.md` — the data contracts every workflow relies on: the
   upstream preparation assumptions, the preprocessed measurement tables,
   and the column-level contracts (case-sensitive) for the feature matrix
   and the clinical cohort table. Read before preparing any input data.

### Second pass: every document, in execution order

Read the `workflow_*.md` files in the order the workflows run, and consult
each entry point in `workflows/` plus its configuration in `configs/`:

| Document | Entry point | Purpose |
|---|---|---|
| `workflow_00.md` | — (preparation contract) | Data assets to prepare before running the workflows |
| `data_schema.md` | — | Column-level input contracts for every table above |
| `workflow_01.md` | `workflows/01_snr_qc.py` | Feature- and sample-level SNR quality control and sensitivity correction |
| `workflow_02.md` | `workflows/02_batch_analysis.py` | Normalization and optional ComBat batch correction of the LC/non-LC matrix |
| `workflow_03.md` | `workflows/03_environmental_voc.py` | Environmental VOC identification and removal |
| `workflow_04.md` | `workflows/04_data_split.py` | Eligibility filtering and discovery/external split construction |
| `workflow_05.md` | `workflows/05_lc_nonlc_analysis.py` | Differential marker analysis (GLMs, AUCs, OPLS-DA) and RFECV feature selection |
| `workflow_06.md` | `workflows/06_lc_nonlc_modelling.py` (`base` / `confounding` modes) | Five-family model comparison with cross-validated predictions and cutoff evaluation; confounder-adjustment modelling (on workflow-05-selected features) |
| `workflow_07.md` | `workflows/07_subgroup_analysis.py` (7 subcommands) | Subgroup analysis, decision curves, SHAP explanation, PR/calibration, prevalence-adjusted PPV/NPV, ROC comparison, confounder models, review scenarios |
| `figure_inputs.md` | — | Figure-generation contracts |

Every executable workflow document follows the same structure: **Purpose**,
**Inputs**, **Execution** (the command contract), **Outputs**, and
**Exclusions** — the known intentional deviations from the source notebook.
The Exclusions section is the fastest way to understand the boundary of each
public workflow. `workflow_00.md` and `data_schema.md` are the exceptions:
contract-only documents with no entry point.

## The Workflow at a Glance

### Execution Order

Upstream data preparation (instrument formatting, clinical eligibility
filtering, and the breath/environmental separation) is assumed complete
before step 1; the preparation contract is defined in `workflow_00.md`
(contract only, no executable entry point) and the column-level contracts
in `data_schema.md`.

1. Apply signal-to-noise quality control and sensitivity correction.
2. Assess and correct batch effects.
3. Analyze environmental VOC measurements.
4. Create and record analysis splits.
5. Perform differential marker analysis.
6. Train and evaluate classification models.
7. Perform confounding-adjusted modelling.
8. Perform model and subgroup analyses.
9. Run the review-response analyses required for the manuscript.

### Migration Principles

- Preserve manuscript parameters, thresholds, sample definitions, and statistical methods.
- Transcribe implicit notebook parameters into versioned configuration.
- Do not retune models or modify reported results as part of code migration.
- Retain a source mapping for every migrated function and workflow.
- Remove private package dependencies by migrating the required implementations into `src/`.
- Use immutable source fields for subgroup membership and cohort assignment.
- Exclude ad hoc record-modification pipelines and all downstream dependencies from the public workflow.

### Execution Status

Workflows 01–07 are migrated, compile-checked, and CLI-verified. Synthetic-data smoke tests exercise every workflow end to end where dependencies are available in the authoring environment; reproducing the reported numbers against the authorized study data requires access to that data. Known intentional deviations from the original analysis are documented in each `docs/workflow_*.md`.

## Conventions

- All input and output paths are caller-supplied; no document refers to an
  internal directory layout. `--output-dir` is a **base** directory: every
  workflow writes into a fixed per-workflow subdirectory, so results always
  land in `{output-dir}/<workflow-name>` (for `07_subgroup_analysis`,
  `{output-dir}/07_subgroup_analysis/<subcommand>`). The default base is
  `./analysis-output`; directories are created when missing.
- Every workflow prints structured run logs: `[i/n] <step>` headers with
  indented sub-conclusions, `tqdm` progress bars inside long-running loops,
  and a final "Final outputs (for downstream use)" section that marks which
  files feed the next workflow.
- Study-specific analysis parameters are versioned in `configs/`
  (`preprocessing.yaml` for SNR QC thresholds, `batches.yaml` for ComBat batch
  assignment rules, `cohort_split.yaml` for the workflow-04 discovery/external
  split). The committed values reproduce the source study; users
  analyzing their own data must review and edit all of them. ComBat correction
  is optional and runs only when a batch config is passed to workflow 02; the
  cohort split is likewise optional and only used when passed via
  `--split-config`.
- Qualitative VOC identification (confirming candidate annotations against
  spectral libraries and metabolic/physiological logic, and excluding
  implausible markers) is a manual step that sits between threshold screening
  and RFECV in workflow 05. It is never automated; its outcome enters as
  workflow 05's `--authorized-marker-table`, and workflow 06 assumes it is
  complete.
- No study data are distributed. Numerical and graphical reproduction
  requires authorized access to the corresponding study data.
- Column names in input contracts are case-sensitive; missing or duplicated
  required fields must produce an explicit error.
- Upstream data preparation (instrument formatting, clinical eligibility
  filtering, and the breath/environmental separation) is assumed complete
  before the workflows run; the preparation contract is `workflow_00.md`.
- Every intentional deviation from a source notebook is documented in the
  Exclusions section of the corresponding `workflow_*.md`.
