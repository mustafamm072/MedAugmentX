"""Showcase of the transforms added in MedAugmentX 0.6.0.

Demonstrates the new general-purpose transforms (cutout, shape normalisation,
adaptive contrast, histogram matching) and the new modality artifacts
(MRI motion, CT metal streak, X-ray scatter/grid, DBT compression/recon
streak) — all seedable and serialisable like every other transform.

Run with:

    python examples/new_transforms.py
"""
from __future__ import annotations

import numpy as np

from medaugmentx import Compose, MedVolume
from medaugmentx.serialization import from_json, to_json
from medaugmentx.transforms import (
    CenterCrop,
    CLAHEContrast,
    CoarseDropout,
    CompressionVariation,
    GridArtifact,
    HistogramMatch,
    MetalStreak,
    MRIMotion,
    Pad,
    ReconStreak,
    Resize,
    ScatterSimulation,
    Sharpen,
)


def main() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((40, 256, 256), dtype=np.float64).astype(np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[10:30, 64:192, 64:192] = 1
    vol = MedVolume(image=image, mask=mask, spacing=(1.0, 0.1, 0.1), metadata={"modality": "DBT"})

    # A reference distribution for histogram matching (a different scan).
    reference = rng.normal(0.4, 0.15, size=(64, 64)).astype(np.float32)

    print("--- General-purpose transforms ---")
    for t in [
        CoarseDropout(num_holes=(2, 6), hole_size=(0.05, 0.15)),
        Sharpen(alpha=(0.3, 0.9)),
        CLAHEContrast(clip_limit=(1.5, 3.0)),
        HistogramMatch(reference=reference, blend=(0.5, 1.0)),
    ]:
        out = t.__class__(**t.to_dict()["params"], seed=0)(vol)
        print(f"  {t.__class__.__name__:18s} shape={out.image.shape} dtype={out.image.dtype}")

    print("\n--- Shape normalisation (resize -> pad/crop to a fixed batch shape) ---")
    target = (48, 224, 224)
    normalise = Compose([Resize(size=(48, 200, 200)), Pad(size=target), CenterCrop(size=target)], seed=0)
    out = normalise(vol)
    print(f"  {vol.image.shape} -> {out.image.shape}  (mask labels preserved: {set(np.unique(out.mask).tolist())})")

    # Intensity artifacts leave the mask untouched; CompressionVariation is a
    # spatial transform, so it warps the mask in lockstep with the image.
    print("\n--- Modality artifacts (intensity artifacts never touch the mask) ---")
    for t in [
        MRIMotion(degrees=(2.0, 5.0)),
        MetalStreak(intensity=(0.15, 0.3)),
        ScatterSimulation(fraction=(0.2, 0.4)),
        GridArtifact(amplitude=(0.04, 0.1)),
        CompressionVariation(scale=(0.85, 1.15)),  # spatial: mask warps too
        ReconStreak(amplitude=(0.1, 0.2)),
    ]:
        out = t.__class__(**t.to_dict()["params"], seed=0)(vol)
        changed = not np.allclose(out.image, vol.image)
        labels_ok = set(np.unique(out.mask).tolist()) <= {0, 1}
        print(f"  {t.__class__.__name__:20s} changed={changed} mask_labels_valid={labels_ok}")

    print("\n--- Everything serialises ---")
    pipeline = Compose(
        [
            CoarseDropout(num_holes=3),
            CLAHEContrast(),
            CompressionVariation(),
            ReconStreak(),
        ],
        seed=7,
    )
    rt = from_json(to_json(pipeline))
    a, b = pipeline(vol), rt(vol)
    print("  JSON round-trip identical:", np.array_equal(a.image, b.image))


if __name__ == "__main__":
    main()
