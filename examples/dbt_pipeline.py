"""End-to-end Phase-1 tomosynthesis (DBT) pipeline.

Demonstrates the four DBT transforms available in Phase 1, with realistic
anisotropic spacing and a single foreground lesion-like region.
"""
from __future__ import annotations

import numpy as np

from medaugment import Compose, MedVolume
from medaugment.transforms import (
    AnisotropicElastic,
    GammaCorrection,
    LimitedAngleBlur,
    RandomFlip,
    SlabShift,
    SliceDropout,
)


def synthesise_dbt_volume(seed: int = 0) -> MedVolume:
    rng = np.random.default_rng(seed)
    # Tiny synthetic DBT slab — anisotropic spacing typical of clinical DBT.
    image = rng.random((40, 384, 256), dtype=np.float64).astype(np.float32) * 0.2
    mask = np.zeros_like(image, dtype=np.uint8)
    # A bright "lesion" centered in the slab.
    image[18:22, 180:204, 120:144] += 0.6
    mask[18:22, 180:204, 120:144] = 1
    return MedVolume(
        image=image,
        mask=mask,
        spacing=(1.0, 0.1, 0.1),  # mm: 10x anisotropy
        metadata={"modality": "DBT", "vendor": "generic"},
    )


def build_dbt_pipeline(seed: int = 0) -> Compose:
    return Compose([
        RandomFlip(axes=("x",), p_per_axis=0.5),
        SlabShift(max_shift=2, p=0.5),
        AnisotropicElastic(alpha=(100.0, 100.0, 8.0), sigma=(8.0, 8.0, 2.0), p=0.4),
        LimitedAngleBlur(arc_degrees=(15.0, 25.0), base_sigma=1.0, p=0.3),
        SliceDropout(num_slices=(0, 2), p=0.2),
        GammaCorrection(gamma=(0.85, 1.15), p=0.4),
    ], seed=seed)


def main() -> None:
    vol = synthesise_dbt_volume()
    pipeline = build_dbt_pipeline(seed=2025)
    out = pipeline(vol)

    print("Input :", vol)
    print("Output:", out)
    print()
    print("Shape preserved   :", out.image.shape == vol.image.shape)
    print("Spacing preserved :", out.spacing == vol.spacing)
    print("Mask label set    :", sorted(set(np.unique(out.mask).tolist())))
    print()
    print("Pipeline:")
    for t in pipeline:
        print(" -", t)


if __name__ == "__main__":
    main()
