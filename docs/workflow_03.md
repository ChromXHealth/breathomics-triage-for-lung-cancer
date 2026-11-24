# Workflow 03: Environmental VOC Analysis for LC/non-LC

## Purpose

Compare environmental and breath measurements by sampling location and remove environmental VOCs and records from the LC/non-LC dataset.

## Inputs

All paths are caller-supplied:

- `--corrected-voc`: workflow 02 LC/non-LC corrected matrix.
- `--breath-locations`: breath sample-location table.
- `--environment-locations`: environmental sample-location table.
- `--voc-info`: VOC annotation workbook.
- `--output-dir`: destination directory.

Source-specific location headers are supplied through `--sample-id-column` and `--location-column`. The two location files are concatenated before standardization. Column contracts for the location tables and the annotation workbook live in `data_schema.md`.

## Execution

Run from this repository's root:

```bash
python -m workflows.03_environmental_voc \
  --corrected-voc /path/to/batch_outputs/combat_voc_LC_nonLC_ENV.csv \
  --breath-locations /path/to/breath_sample_locations.csv \
  --environment-locations /path/to/environment_sample_locations.csv \
  --voc-info /path/to/voc_annotations.xlsx \
  --sample-id-column SampleID \
  --location-column SampleLoc \
  --output-dir /path/to/environmental_outputs
```

## Analysis Rules

- Non-environmental records are labelled BR.
- A site requires both groups and more than three records per group.
- Welch's t-test, within-site false-discovery control, and direction-independent AUC are calculated by VOC.
- ROC classification assigns BR when any site has AUC greater than 0.8; otherwise it assigns ENV.
- FDR classification assigns BR when more than one site has FDR below 0.05; otherwise it assigns ENV.
- VOCs labelled ENV by both rules and all environmental sample records are removed.

Independent LC/HC and LC/LN outputs are intentionally excluded.
