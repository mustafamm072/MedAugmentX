"""Controlled synthetic segmentation ablation. No patient data or clinical claims.

Requires medaugmentx[torch]. Trains identical small CNNs on fixed phantom splits,
compares no augmentation, spatial augmentation, and an MRI appearance policy.
Held-out images use a shifted intensity/noise distribution, not a real scanner.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn

from medaugmentx import Compose, MedVolume, __version__, apply_for_sample
from medaugmentx.transforms import BiasField, GaussianNoise, RandomAffine


def dataset(seed, count, shifted=False):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:32, :32]
    result = []
    for _ in range(count):
        cy, cx = rng.uniform(10, 22, 2)
        ry, rx = rng.uniform(4, 9, 2)
        mask = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1).astype(np.uint8)
        image = gaussian_filter(mask.astype(float), 0.7) * rng.uniform(0.5, 1)
        if shifted:
            image *= 0.55 + 0.7 * xx / 31
            image += rng.uniform(0.05, 0.2)
        image += rng.normal(0, 0.07 if shifted else 0.025, mask.shape)
        result.append(MedVolume(image.astype(np.float32), mask))
    return result


def run(seed, policy_name, train, test, epochs):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 1, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    spatial = [RandomAffine(rotation=15, translation=0.1, scale=(0.9, 1.1))]
    policy = Compose(
        []
        if policy_name == "none"
        else spatial
        + (
            [BiasField(alpha=0.4), GaussianNoise(std=(0.01, 0.1))]
            if policy_name == "appearance"
            else []
        )
    )
    start = time.perf_counter()
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(len(train))
        for offset in range(0, len(order), 8):
            batch = [
                apply_for_sample(policy, train[int(i)], seed=seed, sample_id=int(i), epoch=epoch)
                for i in order[offset : offset + 8]
            ]
            x = torch.from_numpy(np.stack([v.image for v in batch])[:, None])
            y = torch.from_numpy(np.stack([v.mask for v in batch]).astype(np.float32)[:, None])
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        pred = (
            model(torch.from_numpy(np.stack([v.image for v in test])[:, None]))
            .sigmoid()
            .numpy()[:, 0]
            > 0.5
        )
    labels = np.stack([v.mask for v in test]).astype(bool)
    dice = (
        2
        * (pred & labels).sum(axis=(1, 2))
        / np.maximum(pred.sum(axis=(1, 2)) + labels.sum(axis=(1, 2)), 1)
    )
    return {
        "seed": seed,
        "policy_name": policy_name,
        "policy": policy.to_dict(),
        "mean_dice": float(dice.mean()),
        "per_image_dice": dice.tolist(),
        "train_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    if args.epochs < 1 or min(args.seeds) < 0:
        parser.error("epochs must be positive and seeds non-negative")
    torch.set_num_threads(1)
    train, test = dataset(31415, 64), dataset(92653, 48, shifted=True)
    results = [
        run(seed, name, train, test, args.epochs)
        for seed in args.seeds
        for name in ("none", "spatial", "appearance")
    ]
    report = {
        "scope": "Synthetic circle/ellipse segmentation only. Does not establish medical dataset utility or scanner realism.",
        "environment": {
            "medaugmentx": __version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_threads": 1,
        },
        "design": {
            "train_seed": 31415,
            "test_seed": 92653,
            "train_n": 64,
            "test_n": 48,
            "epochs": args.epochs,
            "batch_size": 8,
            "shape": [32, 32],
            "threshold": 0.5,
            "selection": "No tuning or selection on test split",
        },
        "results": results,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    for name in ("none", "spatial", "appearance"):
        values = [r["mean_dice"] for r in results if r["policy_name"] == name]
        print(f"{name}: mean Dice {np.mean(values):.4f}, seed std {np.std(values):.4f}")


if __name__ == "__main__":
    main()
