import os
from pathlib import Path
import subprocess

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_all_workflows_run_on_synthetic_data(tmp_path):
    data_dir = tmp_path / "fixtures"
    output_dir = tmp_path / "outputs"
    env = os.environ.copy()
    env["SYNTHETIC_DATA_DIR"] = str(data_dir)
    env["SYNTHETIC_OUTPUT_DIR"] = str(output_dir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        ["bash", "examples/run_end_to_end.sh"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=420,
    )
    if result.returncode:
        raise AssertionError(
            f"Synthetic workflow failed ({result.returncode}).\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    assert "Smoke run passed" in result.stdout

    discovery = pd.read_csv(output_dir / "04_data_split/VOC_LC_nonLC_discovery.csv")
    external = pd.read_csv(output_dir / "04_data_split/VOC_LC_nonLC_external.csv")
    assert len(discovery) == 80
    assert len(external) == 40
    assert set(discovery["SampleID"]).isdisjoint(external["SampleID"])

    markers = pd.read_csv(output_dir / "05_lc_nonlc_analysis/rfecv_selected_markers.csv")
    assert not markers.empty
    summary = pd.read_csv(output_dir / "06_lc_nonlc_modelling/base/evaluation_summary.csv")
    assert set(summary["model"]) == {"logistic", "random_forest", "knn", "xgboost", "svm"}
