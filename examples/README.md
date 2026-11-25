# Synthetic End-to-End Example

This example verifies the public workflow interfaces without any study data.
All identifiers, measurements, annotations, and clinical fields are generated
from random numbers with a fixed seed. They are not real observations and must
not be used for scientific or clinical interpretation. The separate public
200-sample External Test Set raw-data archive is documented under “Data
Availability” in the repository `README.md`; it is not used by this runner.

## Run

From the repository root, with the reference Conda environment active:

```bash
python examples/generate_synthetic_data.py --output-dir examples/synthetic-data
bash examples/run_end_to_end.sh
```

The runner regenerates the fixtures and executes workflows 01–06 plus all
seven workflow-07 subcommands. It uses two-fold cross-validation and reduced
bootstrap/component-search settings to keep the smoke run practical. ComBat,
SHAP, and all five model families are exercised; slow figure rendering is
skipped.

To use temporary locations:

```bash
SYNTHETIC_DATA_DIR=/tmp/breathomics-fixtures \
SYNTHETIC_OUTPUT_DIR=/tmp/breathomics-outputs \
bash examples/run_end_to_end.sh
```

To select a specific interpreter, set `PYTHON`, for example
`PYTHON=/path/to/python`. Generated directories under `examples/` are ignored
by Git.

## What is checked

The runner verifies key output files from preprocessing, splitting, feature
selection, base/confounding models, subgroup analysis, SHAP, PR/calibration,
prevalence adjustment, ROC comparison, confounder models, and review
scenarios. The automated test suite invokes the same runner in an isolated
temporary directory.

This is an **execution and interface test only**. It does not reproduce the
manuscript cohorts, marker panel, numerical estimates, or figures.
