# Data Schema

## General Rules

No study data are included in this repository. Users must supply authorized local files matching the contracts below. Column names are case-sensitive. Missing or duplicated required fields must produce an explicit error.

Source identifiers, labels, clinical fields, prediction scores, and cohort assignments must not be modified by subgroup analysis or figure-generation code.

## Upstream Data Preparation

The workflows assume the caller has completed three preparation steps before
workflow 01 runs:

1. Instrument output has been converted into sample-by-feature tables: one
   row per sample, with feature columns named by the study's feature
   identifiers.
2. Clinical eligibility filtering has been applied: records without a valid
   LC/non-LC disease label are removed.
3. Breath, environmental, and postoperative follow-up measurements have
   been separated into distinct datasets.

## Preprocessed Measurement Tables

Consumed by workflow 01. Column names are case-sensitive. The VOC and SNR
tables must contain the same samples in the same row order and the same
feature columns.

| Table | Required columns | Notes |
|---|---|---|
| LC/non-LC VOC | `SampleID`, `label`, VOC feature columns | `label` is `LC` or `nonLC`; must include the configured extra feature (if any), which the workflow-01 QC rules retain explicitly |
| LC/non-LC SNR | `SampleID`, `label`, VOC feature columns | Same layout and row order as the LC/non-LC VOC table; SNR values |
| Environmental VOC | `SampleID`, VOC feature columns | No `label`; workflow 01 assigns `ENV` and aligns the features to the QC-retained columns |
| Follow-up VOC | `SampleID`, VOC feature columns | No `label`; workflow 01 assigns `postOp` and aligns the retained features |

## Sampling Location Tables

Consumed by workflow 03 as `--breath-locations` and `--environment-locations`.
One row per sample; the two files are concatenated before standardization.
Header names are overridable through `--sample-id-column` (default
`SampleID`) and `--location-column` (default `SampleLoc`); extra columns are
ignored.

| Column | Type | Description |
|---|---|---|
| `SampleID` | string | Sample identifier, matched against the corrected VOC matrix; unmatched rows are dropped |
| `SampleLoc` | string | Sampling-site code; rows with a missing or `UNK` location are dropped |

At least one site must contain both breath (`BR`) and environmental (`ENV`)
samples with more than three records per group, otherwise workflow 03 stops
with an explicit error.

## VOC Annotation Table

Consumed by workflow 03 as `--voc-info`. The annotation workbook exported by
the instrument software (MS-DIAL class); read from the first worksheet.

| Column | Type | Description |
|---|---|---|
| `Alignment ID` | string or numeric | Unique VOC identifier; renamed to `VOC ID` and joined to the feature columns of the corrected matrix (matched as strings) |
| `Average Rt(min)` | numeric | Retention-time metadata, preserved as float32 |
| `Quant mass` | numeric | Quantifier-mass metadata, preserved as float32 |
| `S/N average` | numeric | Average signal-to-noise metadata, preserved as float32 |

`Alignment ID` must be unique — it is the join key, and duplicated rows would
duplicate results. Additional annotation columns (compound name, formula,
etc.) are preserved into `ENV_BR_joint_label.csv` for manual review; rows
whose identifier is not among the analyzed features are dropped by the join.

## Clinical Label Table

Consumed by workflow 04 as `--clinical-input` and preserved into its cinfo
outputs. One row per participant; upstream eligibility filtering is assumed
complete.

| Column | Type | Description |
|---|---|---|
| `SampleID` | string | Participant identifier |
| `LC_nonLC` | string | `LC` or `nonLC` analysis label |
| `label_postOp` | string | `T` for postoperative follow-up records, which workflow 04 excludes from discovery and external analysis |
| `smoke_hist`, `BMI` | string or numeric | Normalized by workflow 04; `UNK` marks unknown |
| `height`, `weight` | numeric | Optional; dropped by workflow 04 |

## Feature Matrix

The default feature-matrix interface is:

| Column | Type | Description |
|---|---|---|
| `SampleID` | string | De-identified sample identifier |
| `label` | string or integer | Analysis label with an explicitly configured positive class |
| Ordered VOC columns | numeric | Exactly twelve VOC measurements in the configured model order |

Model code must select features by their configured names rather than by their physical positions in the input table. Training, validation, and external inference must use the same feature contract. External inference must not trigger model fitting or threshold recalibration.

## Clinical Cohort Table (cinfo)

Used by workflows 04 (split outputs), 06 confounding mode (covariates), 13
(`confounder-models`), and 07 (subgroups). One row per participant.

| Column | Type | Description |
|---|---|---|
| `SampleID` | string | Participant identifier |
| `label` | string | `LC` or `nonLC` |
| `gender` | string | `M` / `F` |
| `age` | numeric | Age in years |
| `BMI`, `smoke_hist`, `second-hand_smoke_status`, `env_smoke_status` | string or numeric | Categorical confounder fields; an `UNK` level marks unknown |
| `hazardous_substance_exposure`, `family_Ca_hist`, `lung_disease_hist`, `Ca_hist` | string or numeric | Categorical confounder fields |

Subgroup workflows may additionally consume `copd`, `ajcc_stage`,
`nodule_size`, `pathology_subtype`, `nodule_type`, `multiplicity`, and a
`nonlc_subtype` table; workflows document which fields they need.

## Imaging Label Table

Used by `workflows/07_subgroup_analysis.py subgroups` (required) and
`review-scenarios` (optional, enables the AJCC stage-I scenario table). One
row per sample; the caller prepares this table from the study's imaging and
pathology records with the column names and value vocabularies below —
cohort-specific source names and languages are renamed before calling, the
workflow only accepts this contract.

| Column | Required | Value vocabulary |
|---|---|---|
| `SampleID` | yes | Sample identifier, joinable to the prediction tables |
| `ajcc_stage` | one of the five | `0`, `IA1`, `IA2`, `IA3`, `IB`, `II`, `III`, `IV`; anything else (including `UNK` and missing) is excluded from stage groups |
| `nodule_size` | one of the five | `nodule_size <= 10`, `10 < nodule_size <= 20`, `20 < nodule_size <= 30`, `30 < nodule_size` (ASCII `<=`) |
| `pathology_subtype` | one of the five | `AIS`, `MIA`, `IA`, `SCC`, `SCLC`; anything else is excluded from subtype groups |
| `nodule_type` | one of the five | `GGN`, `PSN`, `SN` |
| `multiplicity` | one of the five | `SPN`, `MPN` |

All five label columns are optional individually — dimensions whose column
is absent are skipped — but the table must carry at least `SampleID`.

## Lab Records Table

Used by `workflows/07_subgroup_analysis.py review-scenarios`. The
collection-to-analysis interval analysis was removed, so no timestamp
columns are required.

| Column | Type | Description |
|---|---|---|
| `SampleID` | string | Sample identifier (the `#`-prefix segment defines the scenario) |
| `dset` | string | Dataset name |
| `label` | integer 0/1 or `LC`/`nonLC` | Reference label (1 = LC) |
| `site` | string | Sampling site (optional, enables site counts) |

## Prediction Table

The standard prediction interface contains:

| Column | Type | Description |
|---|---|---|
| `SampleID` | string | Identifier used for traceable joins |
| `label` | string or integer | Reference label |
| `pred` | float | Model score or probability for the positive class |

`roc-compare` consumes one table with `label` plus one probability column
per model, named via `--model NAME=COLUMN`. NELSON predictions use a
`NELSON_FLAG` score column.

Joins must be one-to-one unless a workflow explicitly documents another cardinality. Duplicate identifiers and unmatched records must be reported rather than silently discarded.

## Figure Inputs

Figure scripts consume analysis outputs matching the contracts above. Displayed statistics must be derived from the same records used to draw the corresponding curves or summaries. Manual statistic overrides are disabled by default.
