"""Paired segmentation policy comparisons on a fixed patient holdout."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from medaugmentx.reproducibility import environment_versions


def compare_segmentation_policies(
    references: Mapping[str, np.ndarray],
    predictions: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    *,
    case_to_patient: Mapping[str, str],
    train_patients: Iterable[str],
    labels: Sequence[int],
    validation_patients: Iterable[str] = (),
    baseline: str = "baseline",
    bootstrap_replicates: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare hard segmentation predictions on identical held-out patients.

    Predictions use ``policy name -> training seed -> case ID -> label array``.
    Each policy must cover every case and have the same training seeds as the
    baseline. Arrays must be integer or Boolean 2D/3D maps on the reference
    grid. The caller supplies patient IDs for leakage checks across splits.

    Each requested label contributes Dice in [0, 1]. Two empty masks score 1,
    while a prediction or reference alone scores 0. Average cases within a
    patient, then weight patients, labels and training seeds equally. Paired
    intervals bootstrap patients after averaging across the matched seeds.
    These intervals describe holdout variation conditional on those trained
    runs and require at least two held-out patients. Use zero replicates to
    disable them. The report contains aggregate counts and scores.

    Args:
        references: Ground-truth label maps keyed by held-out case ID.
        predictions: Model label maps by policy, training seed and case ID.
        case_to_patient: Stable patient ID for every held-out case.
        train_patients: Patient IDs used to train any compared policy.
        labels: Distinct nonzero integer class IDs to evaluate.
        validation_patients: Patient IDs used for tuning or model selection.
        baseline: Name of the unaugmented policy in ``predictions``.
        bootstrap_replicates: Number of patient bootstrap draws for 95% intervals.
        seed: Non-negative bootstrap RNG seed.
    """
    if not references or references.keys() != case_to_patient.keys():
        raise ValueError("case_to_patient must cover exactly the non-empty reference case set")
    if any(not isinstance(case, str) or not case.strip() for case in references):
        raise ValueError("case IDs must be non-empty strings")
    train = _patient_ids(train_patients)
    validation = _patient_ids(validation_patients)
    holdout = _patient_ids(case_to_patient.values())
    if not train:
        raise ValueError("train_patients must include at least one patient")
    if train & validation or train & holdout or validation & holdout:
        raise ValueError("Patient overlap between training, validation and holdout splits")

    class_ids = tuple(labels)
    if not class_ids or any(not _integer(label) or label == 0 for label in class_ids):
        raise ValueError("labels must contain nonzero integer IDs")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("labels must be distinct")
    if any(not _integer(n) or n < 0 for n in (bootstrap_replicates, seed)):
        raise ValueError("bootstrap_replicates and seed must be non-negative integers")
    if baseline not in predictions or len(predictions) < 2:
        raise ValueError("predictions must include the baseline and at least one other policy")

    seeds = predictions[baseline].keys()
    if not seeds or any(not _integer(n) or n < 0 for n in seeds):
        raise ValueError("training seeds must be non-negative integers")
    for name, runs in predictions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("policy names must be non-empty strings")
        if runs.keys() != seeds or any(not _integer(n) for n in runs):
            raise ValueError("Each policy must use exactly the same training seeds as the baseline")
        if any(case_predictions.keys() != references.keys() for case_predictions in runs.values()):
            raise ValueError("Every policy run must predict exactly the held-out case set")

    ordered_seeds = sorted(seeds)
    patient_rows = {patient: i for i, patient in enumerate(sorted(holdout))}
    patients = len(patient_rows)
    counts = np.zeros(patients, dtype=np.intp)
    scores = {
        name: np.zeros((len(ordered_seeds), patients, len(class_ids)), dtype=np.float64)
        for name in predictions
    }
    reference_support = np.zeros(len(class_ids), dtype=np.intp)
    for case in sorted(references):
        expected = references[case]
        _check_label_map(expected)
        row = patient_rows[case_to_patient[case]]
        counts[row] += 1
        pairs = []
        for name, runs in predictions.items():
            for index, training_seed in enumerate(ordered_seeds):
                actual = runs[training_seed][case]
                _check_label_map(actual)
                if actual.shape != expected.shape:
                    raise ValueError("Every prediction shape must match its reference")
                pairs.append((name, index, actual))

        for col, label in enumerate(class_ids):
            target = _label_mask(expected, int(label))
            target_count = np.count_nonzero(target)
            reference_support[col] += target_count > 0
            for name, index, actual in pairs:
                positive = _label_mask(actual, int(label))
                denominator = target_count + np.count_nonzero(positive)
                dice = 2 * np.count_nonzero(positive & target) / denominator if denominator else 1.0
                scores[name][index, row, col] += dice

    by_patient = {name: matrix / counts[None, :, None] for name, matrix in scores.items()}
    baseline_dice = by_patient[baseline]
    results = {}
    for name, matrix in by_patient.items():
        seed_dice = matrix.mean(axis=(1, 2))
        paired_deltas = (matrix - baseline_dice).mean(axis=(0, 2))
        interval = None
        if name != baseline and bootstrap_replicates and patients >= 2:
            rng = np.random.default_rng(int(seed))
            draws = [
                paired_deltas[rng.integers(patients, size=patients)].mean()
                for _ in range(bootstrap_replicates)
            ]
            interval = np.percentile(draws, (2.5, 97.5)).tolist()
        results[name] = {
            "mean_dice": float(seed_dice.mean()),
            "dice_by_seed": seed_dice.tolist(),
            "seed_std": float(seed_dice.std()),
            "dice_by_label": matrix.mean(axis=(0, 1)).tolist(),
            "delta_vs_baseline": float(paired_deltas.mean()),
            "patient_bootstrap_ci95": interval,
        }
    return {
        "environment": environment_versions(),
        "baseline": baseline,
        "training_seeds": [int(n) for n in ordered_seeds],
        "labels": [int(label) for label in class_ids],
        "holdout_cases": len(references),
        "holdout_patients": patients,
        "cases_with_label": reference_support.tolist(),
        "bootstrap_seed": int(seed),
        "bootstrap_replicates": int(bootstrap_replicates),
        "policies": results,
    }


def _integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _patient_ids(values: Iterable[str]) -> set[str]:
    if isinstance(values, str):
        raise ValueError("Patient IDs must be supplied as an iterable of IDs")
    ids = set(values)
    if any(not isinstance(patient, str) or not patient.strip() for patient in ids):
        raise ValueError("Patient IDs must be non-empty strings")
    return ids


def _check_label_map(array: np.ndarray) -> None:
    if not isinstance(array, np.ndarray) or array.dtype.kind not in "bui":
        raise ValueError("Evaluation requires integer or Boolean NumPy label maps")
    if array.ndim not in (2, 3) or not array.size:
        raise ValueError("Evaluation requires non-empty 2D or 3D label maps")


def _label_mask(array: np.ndarray, label: int) -> np.ndarray:
    if array.dtype.kind == "b":
        return array if label == 1 else np.zeros(array.shape, dtype=bool)
    bounds = np.iinfo(array.dtype)
    if not int(bounds.min) <= label <= int(bounds.max):
        return np.zeros(array.shape, dtype=bool)
    return array == np.asarray(label, dtype=array.dtype)
