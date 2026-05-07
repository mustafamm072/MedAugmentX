"""Hello-world MedAugment pipeline.

Run with:

    python examples/quickstart.py
"""
from __future__ import annotations

import numpy as np

from medaugment import Compose, MedVolume, OneOf
from medaugment.transforms import (
    ElasticDeform,
    GammaCorrection,
    GaussianNoise,
    RandomAffine,
    RandomFlip,
    RicianNoise,
)


def main() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((40, 128, 128), dtype=np.float64).astype(np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[10:30, 32:96, 32:96] = 1

    vol = MedVolume(image=image, mask=mask, spacing=(1.0, 0.7, 0.7), metadata={"modality": "MR"})

    pipeline = Compose([
        RandomFlip(axes=("x",), p_per_axis=0.5),
        RandomAffine(rotation=15.0, scale=(0.95, 1.05), p=0.7),
        ElasticDeform(alpha=(60.0, 60.0, 5.0), sigma=(6.0, 6.0, 2.0), p=0.5),
        OneOf([RicianNoise(std=0.02), GaussianNoise(std=0.015)], p=0.6),
        GammaCorrection(gamma=(0.85, 1.15), p=0.5),
    ], seed=42)

    out = pipeline(vol)

    print("Input :", vol)
    print("Output:", out)
    print("Pipeline shape preserved:", out.image.shape == vol.image.shape)
    print("Mask labels preserved   :", set(np.unique(out.mask).tolist()))


if __name__ == "__main__":
    main()
