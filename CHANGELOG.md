# Changelog

## v1.0.2-rc — 2026-11-25

### Documentation and release metadata

- Updated the manuscript and repository title to *Pre-imaging stratification
  for lung cancer detection based on breath exhalate*.
- Updated the current development version to `v1.0.2-rc`.
- Positioned GitHub release notes to introduce the repository first, with a
  concise changes section and a pointer to this changelog.
- Aligned the README performance summary with the latest manuscript,
  including the sensitivity- and specificity-prioritized cutoffs.
- Added a documentation fallback stating that the manuscript is authoritative
  if any study description or numerical value differs.

## v1.0.1-rc — 2026-11-25

### Documentation and release metadata

- Clarified the distinction between public execution tests and numerical
  reproduction with authorized study data.
- Documented the tested platform: macOS 27.2.1 (26B131), Python 3.9.18, and
  the `breath-voc-analysis` Conda environment.
- Added installation, optional synthetic checks, data-access, citation, and
  reproducibility guidance.
- Documented the public 200-sample External Test Set raw-data archive and its
  DOI, while clarifying review-and-approval access to controlled full data,
  patient labels, clinical records, and trained models.
- Switched the README badge to the Zenodo concept DOI and aligned the
  development version to `v1.0.1-rc`.
- Added example schemas for study-specific cohort and reviewed-marker assets.

### Analysis code

- Made random seeds explicit across model fitting, feature selection, and
  bootstrap analyses.
- Applied discovery-derived age scaling to the external model table.
- Strengthened required-column and identifier validation.

### Verification and maintenance

- Added synthetic input generation, an end-to-end smoke runner, automated
  tests, Ruff/pytest configuration, and continuous integration.
- Renamed and expanded `requirements.txt` with the resolved direct and
  transitive runtime dependency closure; test tools remain development-only.

No participant-level data, clinical labels, trained model artifacts, or
source-study confidential configuration were added.
