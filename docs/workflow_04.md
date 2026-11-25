# Workflow 04: LC/non-LC Dataset Splitting

## Purpose

Align environmental-VOC-free measurements with clinical records, apply eligibility filters, normalize selected clinical fields, and construct discovery and external datasets.

## Inputs

All paths and cohort-selection values are caller-supplied:

- `--voc-input`: workflow 03 `evoc_free_LC_nonLC.csv` table or an equivalent authorized table.
- `--clinical-input`: clinical-label table with `SampleID`, `LC_nonLC`, `label_postOp`, `smoke_hist`, and `BMI` (upstream eligibility filtering assumed complete; see `data_schema.md`).
- `--output-dir`: destination directory.
- Cohort selection, either through a `--split-config` YAML file or through
  repeated `--discovery-project`, `--external-project`,
  `--discovery-id-prefix`, and `--external-id-prefix` values (mutually
  exclusive). Without a config, both project lists are required; the
  id-prefix lists are optional and select cohorts split inside one project
  (matched against the start of `SampleID`).

Repeat project arguments for every included study. Identifier-prefix arguments preserve special cohort splits based on the start of `SampleID`. No study-specific project names are embedded in the public code. `configs/cohort_split.example.yaml` documents the schema only and is not the source-study split. Numerical reproduction requires an authorized study-specific config; no split file is loaded by default.

## Execution

Run from this repository's root:

```bash
python -m workflows.04_data_split \
  --voc-input /path/to/environmental_outputs/evoc_free_LC_nonLC.csv \
  --clinical-input /path/to/clinical_labels.csv \
  --output-dir /path/to/split_outputs \
  --discovery-project PROJECT_A \
  --discovery-project PROJECT_B \
  --external-project PROJECT_C \
  --discovery-id-prefix 'PROJECT_D#0' \
  --external-id-prefix 'PROJECT_C#1'
```

or, equivalently, with a cohort-split config:

```bash
python -m workflows.04_data_split \
  --voc-input /path/to/environmental_outputs/evoc_free_LC_nonLC.csv \
  --clinical-input /path/to/clinical_labels.csv \
  --output-dir /path/to/split_outputs \
  --split-config /path/to/cohort_split.yaml
```

## Processing Rules

- Measurements and clinical records are aligned by `SampleID`.
- Records failing upstream quality control are excluded through the alignment.
- Postoperative records are excluded from discovery and external analysis.
- Height and weight are removed while BMI is retained.
- Smoking history is normalized to T, F, or UNK without imputation.
- BMI is categorized using boundaries 18.5, 24, and 28.
- Discovery and external tables are constructed independently from configured projects and identifier prefixes.
- Clinical outputs retain records whose identifiers occur in the corresponding measurement output.

Independent LC/HC and LC/LN splitting branches are intentionally excluded.
