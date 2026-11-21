import numpy as np
import pytest

from medaugmentx import compare_segmentation_policies


def comparison_inputs():
    labels = np.ones((2, 2), dtype=np.uint8)
    missing = np.zeros_like(labels)
    return {
        "references": dict(a=labels, b=labels, c=labels),
        "predictions": {
            "baseline": {4: dict(a=labels, b=labels, c=missing)},
            "augmented": {4: dict(a=labels, b=missing, c=labels)},
        },
        "case_to_patient": {"a": "patient-a", "b": "patient-a", "c": "patient-b"},
        "train_patients": ["patient-train"],
        "validation_patients": ["patient-validation"],
        "labels": [1],
        "seed": 3,
        "bootstrap_replicates": 100,
    }


def test_paired_policy_comparison_weights_patients_equally_and_is_deterministic():
    inputs = comparison_inputs()
    report = compare_segmentation_policies(**inputs)
    assert compare_segmentation_policies(**inputs) == report
    assert report["holdout_cases"] == 3
    assert report["holdout_patients"] == 2
    assert report["cases_with_label"] == [3]
    assert report["policies"]["baseline"]["mean_dice"] == 0.5
    assert report["policies"]["augmented"]["mean_dice"] == 0.75
    assert report["policies"]["augmented"]["delta_vs_baseline"] == 0.25
    assert report["policies"]["augmented"]["patient_bootstrap_ci95"] == [-0.5, 1.0]
    assert "patient-a" not in str(report)


def test_dice_compares_exact_wide_ids_and_sets_no_ci_for_one_patient():
    label = 2**64 - 2
    ground_truth = np.array([[label, 0], [0, 0]], dtype=np.uint64)
    wrong = np.array([[label + 1, 0], [0, 0]], dtype=np.uint64)
    report = compare_segmentation_policies(
        {"x": ground_truth},
        {"baseline": {1: {"x": wrong}}, "augmented": {1: {"x": ground_truth}}},
        case_to_patient={"x": "test"},
        train_patients=["train"],
        labels=[label, 1],
    )
    assert report["policies"]["baseline"]["dice_by_label"] == [0, 1]
    assert report["policies"]["augmented"]["dice_by_label"] == [1, 1]
    assert report["policies"]["augmented"]["patient_bootstrap_ci95"] is None


@pytest.mark.parametrize("patients", ["train_patients", "validation_patients"])
def test_policy_comparison_rejects_patient_leakage(patients):
    inputs = comparison_inputs()
    inputs[patients] = ["patient-a"]
    with pytest.raises(ValueError, match="Patient overlap"):
        compare_segmentation_policies(**inputs)


def test_policy_comparison_rejects_unmatched_seeds_or_cases():
    inputs = comparison_inputs()
    augmented = inputs["predictions"]["augmented"]
    augmented[5] = augmented.pop(4)
    with pytest.raises(ValueError, match="same training seeds"):
        compare_segmentation_policies(**inputs)
    augmented[4] = augmented.pop(5)
    del augmented[4]["b"]
    with pytest.raises(ValueError, match="held-out case set"):
        compare_segmentation_policies(**inputs)


@pytest.mark.parametrize(
    "prediction, matching",
    [(np.ones((2, 2), dtype=float), "integer or Boolean"), (np.ones((3, 2), bool), "shape")],
)
def test_policy_comparison_rejects_invalid_prediction_data(prediction, matching):
    inputs = comparison_inputs()
    inputs["predictions"]["augmented"][4]["a"] = prediction
    with pytest.raises(ValueError, match=matching):
        compare_segmentation_policies(**inputs)
