"""Train and compare two policies on paired synthetic MRI-like channels."""

from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import gaussian_filter

from medaugmentx import Compose, MedVolume, apply_for_sample, compare_segmentation_policies
from medaugmentx.serialization import policy_document
from medaugmentx.transforms import RandomAffine


def phantoms(seed, count, *, held_out=False):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:24, :24]
    affine = np.array([[0, 0.9, 0, -12], [1.1, 0, 0, -12], [0, 0, 1, 5], [0, 0, 0, 1]])
    cases = []
    for _ in range(count):
        cy, cx = rng.uniform(8, 16, size=2)
        ry, rx = rng.uniform(3, 6, size=2)
        mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
        blurred = gaussian_filter(mask.astype(float), 0.8)
        t1 = 0.25 + rng.uniform(0.25, 0.45) * blurred + rng.normal(0, 0.06, mask.shape)
        flair = 0.1 + 0.9 * blurred + rng.normal(0, 0.08, mask.shape)
        if held_out:
            t1 = t1 * (0.85 + xx / 80)
            flair = flair * rng.uniform(0.65, 1.05) + 0.06
        cases.append(
            MedVolume(t1.astype(np.float32), mask, affine=affine).with_channels(
                {"flair": MedVolume(flair.astype(np.float32), affine=affine)}
            )
        )
    return cases


def features(volume):
    channels = volume.stack_channels("flair").reshape(2, -1).T
    return np.column_stack((np.ones(len(channels)), channels))


def train_model(train, policy, seed):
    draws = np.random.default_rng(seed).integers(len(train), size=2 * len(train))
    patches = [
        apply_for_sample(policy, train[i], seed=seed, sample_id=int(i), augmentation_index=j)
        for j, i in enumerate(draws)
    ]
    x = np.concatenate([features(volume) for volume in patches])
    y = np.concatenate([volume.mask.ravel() for volume in patches])
    return np.linalg.solve(x.T @ x + 0.1 * np.eye(x.shape[1]), x.T @ y)


def main():
    train = phantoms(107, 20)
    holdout = {f"scan-{i}": volume for i, volume in enumerate(phantoms(803, 12, held_out=True))}
    policies = {
        "baseline": Compose([]),
        "joint_affine": Compose([RandomAffine(rotation=12, scale=(0.9, 1.1), translation=0.08)]),
    }
    predictions = {}
    for name, policy in policies.items():
        predictions[name] = {}
        for seed in (0, 1, 2):
            weights = train_model(train, policy, seed)
            predictions[name][seed] = {
                case: ((features(volume) @ weights) >= 0.5).reshape(volume.shape)
                for case, volume in holdout.items()
            }
    report = compare_segmentation_policies(
        {case: volume.mask for case, volume in holdout.items()},
        predictions,
        case_to_patient={case: f"holdout-{i}" for i, case in enumerate(holdout)},
        train_patients=[f"train-{i}" for i in range(len(train))],
        labels=(1,),
    )
    report["scope"] = "Synthetic two-channel ellipse segmentation"
    report["policy_documents"] = {
        name: policy_document(policy) for name, policy in policies.items()
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
