"""Hello-world MedAugment pipeline.

Run with:

    python examples/quickstart.py
"""
from __future__ import annotations

import json

import numpy as np

from medaugmentx import Compose, MedVolume, OneOf
from medaugmentx.serialization import from_json, to_json
from medaugmentx.transforms import (
    BiasField,
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
        RandomAffine(rotation=10.0, scale=(0.9, 1.1), translation=(-0.05, 0.05), p=0.7),
        ElasticDeform(alpha=30.0, sigma=4.0, p=0.5),
        BiasField(alpha=0.3, p=0.7),
        OneOf([
            RicianNoise(std=(0.005, 0.02)),
            GaussianNoise(std=0.015),
        ], p=0.6),
        GammaCorrection(gamma=(0.85, 1.15), p=0.5),
    ], seed=42)

    out = pipeline(vol)

    print("Input :", vol)
    print("Output:", out)
    print("Shape preserved:", out.image.shape == vol.image.shape)
    print("Mask labels    :", set(np.unique(out.mask).tolist()))

    # --- Serialisation round-trip ---
    json_str = to_json(pipeline)
    pipeline2 = from_json(json_str)
    out2 = pipeline2(vol)
    print("\nPipeline JSON (first 120 chars):", json_str[:120], "...")
    print("Round-trip identical:", np.array_equal(out.image, out2.image))

    # Pretty-print the pipeline dict
    print("\nPipeline structure:")
    print(json.dumps(pipeline.to_dict(), indent=2, default=str)[:400], "...")


if __name__ == "__main__":
    main()
