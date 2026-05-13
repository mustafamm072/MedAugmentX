"""End-to-end Digital Breast Tomosynthesis (DBT) augmentation pipeline.

Demonstrates the full DBT transform set — anisotropic spatial deformations,
slice-level dropouts, limited-angle blur, and intensity augmentations —
with realistic anisotropic spacing and a synthetic lesion-like region.

Run with:

    python examples/dbt_pipeline.py
"""
from __future__ import annotations

import numpy as np

from medaugmentx import Compose, MedVolume
from medaugmentx.presets import dbt_pipeline
from medaugmentx.transforms import (
    AnisotropicElastic,
    BiasField,
    GammaCorrection,
    LimitedAngleBlur,
    RandomAffine,
    RandomFlip,
    SlabShift,
    SliceDropout,
)


def synthesise_dbt_volume(seed: int = 0) -> MedVolume:
    rng = np.random.default_rng(seed)
    # Tiny synthetic DBT slab — anisotropic spacing typical of clinical DBT.
    image = rng.random((40, 384, 256), dtype=np.float64).astype(np.float32) * 0.2
    mask = np.zeros_like(image, dtype=np.uint8)
    # A bright "lesion" centred in the slab.
    image[18:22, 180:204, 120:144] += 0.6
    mask[18:22, 180:204, 120:144] = 1
    return MedVolume(
        image=image,
        mask=mask,
        spacing=(1.0, 0.1, 0.1),  # mm: 10× anisotropy along Z
        metadata={"modality": "DBT", "vendor": "generic"},
    )


def build_dbt_pipeline(seed: int = 0) -> Compose:
    """Manual pipeline — tune transforms and parameters for your dataset."""
    return Compose([
        RandomFlip(axes=("x",), p_per_axis=0.5),
        RandomAffine(rotation=5.0, scale=(0.95, 1.05), axes_enabled=("x", "y"), p=0.7),
        AnisotropicElastic(alpha=(100.0, 100.0, 8.0), sigma=(8.0, 8.0, 2.0), p=0.4),
        SlabShift(max_shift=2, p=0.5),
        LimitedAngleBlur(arc_degrees=(15.0, 25.0), base_sigma=1.0, p=0.3),
        SliceDropout(num_slices=(0, 2), p=0.2),
        BiasField(alpha=0.2, coarse_shape=3, p=0.5),
        GammaCorrection(gamma=(0.85, 1.15), p=0.4),
    ], seed=seed)


def main() -> None:
    vol = synthesise_dbt_volume()

    # Option A: pre-built preset (recommended starting point)
    preset = dbt_pipeline(seed=0)
    out_preset = preset(vol)
    print("=== Preset pipeline ===")
    print("Input :", vol)
    print("Output:", out_preset)

    # Option B: manually assembled pipeline
    pipeline = build_dbt_pipeline(seed=2025)
    out = pipeline(vol)
    print("\n=== Manual pipeline ===")
    print("Input :", vol)
    print("Output:", out)
    print()
    print("Shape preserved   :", out.image.shape == vol.image.shape)
    print("Spacing preserved :", out.spacing == vol.spacing)
    print("Mask label set    :", sorted(set(np.unique(out.mask).tolist())))
    print()
    print("Pipeline steps:")
    for t in pipeline:
        print(" -", t)


if __name__ == "__main__":
    main()
