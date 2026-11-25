import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from sklearn.ensemble import RandomForestClassifier

from src.calibration import compute_brier_score, compute_calibration_curve
from src.modelling import cross_validated_predictions


def test_calibration_bootstrap_is_reproducible():
    labels = np.array([0, 1] * 10)
    scores = np.linspace(0.02, 0.98, 20)
    first = compute_calibration_curve(labels, scores, n_bootstrap=20, random_state=7)
    second = compute_calibration_curve(labels, scores, n_bootstrap=20, random_state=7)
    for key in ("ci_lower", "ci_upper"):
        assert_allclose(first[key], second[key], equal_nan=True)

    first_brier = compute_brier_score(labels, scores, n_bootstrap=20, random_state=7)
    second_brier = compute_brier_score(labels, scores, n_bootstrap=20, random_state=7)
    assert first_brier["brier_ci_lower"] == second_brier["brier_ci_lower"]
    assert first_brier["brier_ci_upper"] == second_brier["brier_ci_upper"]


def test_seeded_cross_validated_predictions_are_reproducible():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    frame.insert(0, "label", ["LC", "nonLC"] * 20)
    frame.insert(0, "SampleID", [f"S{i}" for i in range(40)])
    parameters = {"n_estimators": 10, "random_state": 11}
    first = cross_validated_predictions(
        RandomForestClassifier, parameters, frame, cv=2, random_state=13
    )
    second = cross_validated_predictions(
        RandomForestClassifier, parameters, frame, cv=2, random_state=13
    )
    assert_allclose(first[0]["pred"], second[0]["pred"])
    assert_allclose(first[1]["pred"], second[1]["pred"])
