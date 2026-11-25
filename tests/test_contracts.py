import numpy as np
import pandas as pd
import pytest

from src.environmental_voc import add_voc_information
from src.evaluation import require_columns, require_unique_sample_ids
from src.modelling import (
    LogisticRegression,
    apply_random_state,
    cross_validated_predictions,
    external_covariates,
)
from src.splitting import split_discovery_and_external


def test_required_columns_rejects_duplicates():
    frame = pd.DataFrame([[1, 2, 3]], columns=["SampleID", "label", "label"])
    with pytest.raises(ValueError, match="Duplicated required columns: label"):
        require_columns(frame, ["SampleID", "label"])


def test_sample_ids_must_be_unique_and_nonmissing():
    duplicated = pd.DataFrame({"SampleID": ["A", "A"]})
    with pytest.raises(ValueError, match="must be unique"):
        require_unique_sample_ids(duplicated)
    missing = pd.DataFrame({"SampleID": ["A", np.nan]})
    with pytest.raises(ValueError, match="missing"):
        require_unique_sample_ids(missing)


def test_external_age_uses_discovery_scaling():
    discovery = pd.DataFrame({"SampleID": ["D1", "D2"], "age": [40, 60], "gender": ["M", "F"]})
    external = pd.DataFrame({"SampleID": ["E1", "E2"], "age": [50, 70], "gender": ["F", "M"]})
    result = external_covariates(external, discovery)
    assert result["age"].tolist() == pytest.approx([0.0, 2.0])
    assert result["gender"].tolist() == [0.0, 1.0]


def test_independent_feature_order_is_enforced():
    train = pd.DataFrame(
        {
            "SampleID": [f"D{i}" for i in range(8)],
            "label": ["LC", "nonLC"] * 4,
            "a": np.arange(8),
            "b": np.arange(8) + 1,
        }
    )
    external = train.rename(columns={"SampleID": "ignored"}).copy()
    external.insert(0, "SampleID", [f"E{i}" for i in range(8)])
    external = external[["SampleID", "label", "b", "a"]]
    with pytest.raises(ValueError, match="feature columns and order"):
        cross_validated_predictions(
            LogisticRegression,
            {},
            train,
            cv=2,
            independent_frames=(external,),
        )


def test_cohort_overlap_is_rejected():
    voc = pd.DataFrame({"SampleID": ["P#1", "P#2"], "label": ["LC", "nonLC"]})
    clinical = pd.DataFrame({"SampleID": ["P#1", "P#2"], "label": ["LC", "nonLC"]})
    with pytest.raises(ValueError, match="overlap"):
        split_discovery_and_external(voc, clinical, ["P"], ["P"], [], [])


def test_annotation_identifiers_must_be_unique():
    joint = pd.DataFrame({"VOC ID": ["1"], "ROCAUC": ["BR"], "FDR": ["BR"]})
    annotation = pd.DataFrame(
        {
            "Alignment ID": [1, 1],
            "Average Rt(min)": [1.0, 2.0],
            "Quant mass": [50, 60],
            "S/N average": [20, 30],
        }
    )
    with pytest.raises(ValueError, match="must be unique"):
        add_voc_information(joint, annotation)


def test_model_seed_default_and_explicit_override():
    seeded = apply_random_state("random_forest", {}, 7)
    assert seeded["random_state"] == 7
    explicit = apply_random_state("random_forest", {"random_state": 99}, 7)
    assert explicit["random_state"] == 99
    assert "random_state" not in apply_random_state("knn", {}, 7)
