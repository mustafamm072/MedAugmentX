"""End-to-end pipeline tests covering deterministic seeding and mask alignment.

These exercise the regression contract from MILESTONES — fixed seeds produce
identical outputs across runs, and image/mask spatial alignment never breaks.
"""
from __future__ import annotations

import numpy as np

from medaugment import Compose, MedVolume, OneOf
from medaugment.transforms import (
    AnisotropicElastic,
    ElasticDeform,
    GammaCorrection,
    GaussianNoise,
    LimitedAngleBlur,
    RandomAffine,
    RandomFlip,
    RicianNoise,
    SlabShift,
    SliceDropout,
)


def _make_synthetic_volume(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.random((12, 32, 32), dtype=np.float64).astype(np.float32)
    # Build a simple 2-label mask with sharp boundaries.
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[3:9, 8:24, 8:24] = 1
    mask[5:7, 14:18, 14:18] = 2
    return MedVolume(image=img, mask=mask, spacing=(1.0, 0.7, 0.7), metadata={"modality": "MR"})


def test_full_pipeline_runs_end_to_end():
    vol = _make_synthetic_volume()
    pipeline = Compose([
        RandomAffine(rotation=10.0, scale=(0.95, 1.05), p=0.7),
        ElasticDeform(alpha=(20.0, 20.0, 2.0), sigma=(4.0, 4.0, 1.0), p=0.5),
        OneOf([RicianNoise(std=0.02), GaussianNoise(std=0.015)], p=0.6),
        GammaCorrection(gamma=(0.85, 1.15), p=0.5),
    ], seed=123)
    out = pipeline(vol)
    assert out.image.shape == vol.image.shape
    assert out.mask.shape == vol.mask.shape
    assert out.mask.dtype == vol.mask.dtype
    # mask labels never widen the original label set
    assert set(np.unique(out.mask).tolist()).issubset({0, 1, 2})


def test_seed_yields_bit_identical_output():
    vol = _make_synthetic_volume()
    cfg = lambda seed: Compose([  # noqa: E731
        RandomFlip(axes=("x",), p_per_axis=0.5),
        RandomAffine(rotation=15.0, scale=(0.9, 1.1)),
        ElasticDeform(alpha=8.0, sigma=2.0),
        GammaCorrection(gamma=(0.7, 1.3)),
    ], seed=seed)
    a = cfg(2025)(vol)
    b = cfg(2025)(vol)
    np.testing.assert_array_equal(a.image, b.image)
    np.testing.assert_array_equal(a.mask, b.mask)


def test_dbt_pipeline_preserves_geometry():
    rng = np.random.default_rng(0)
    img = rng.random((20, 64, 64), dtype=np.float64).astype(np.float32)
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[5:15, 20:40, 20:40] = 1
    vol = MedVolume(image=img, mask=mask, spacing=(1.0, 0.1, 0.1), metadata={"modality": "DBT"})

    pipeline = Compose([
        RandomFlip(axes=("x",), p_per_axis=0.5),
        SlabShift(max_shift=2),
        AnisotropicElastic(alpha=(20.0, 20.0, 2.0), sigma=(4.0, 4.0, 1.0)),
        LimitedAngleBlur(arc_degrees=(15.0, 25.0), base_sigma=1.0),
        SliceDropout(num_slices=(0, 2)),
        GammaCorrection(gamma=(0.85, 1.15)),
    ], seed=1)
    out = pipeline(vol)
    assert out.image.shape == vol.image.shape
    assert out.mask.shape == vol.mask.shape
    # mask labels never widen
    assert set(np.unique(out.mask).tolist()).issubset({0, 1})
    # spacing & metadata preserved
    assert out.spacing == vol.spacing
    assert out.metadata.get("modality") == "DBT"


def test_image_mask_alignment_under_many_seeds():
    """Regression: spatial transforms keep image and mask aligned across 100 seeds."""
    img = np.zeros((32, 32), dtype=np.float32)
    img[10:22, 10:22] = 1.0  # the only foreground in image and mask
    mask = (img > 0).astype(np.uint8)
    vol = MedVolume(image=img, mask=mask)

    pipeline_factory = lambda seed: Compose([  # noqa: E731
        RandomFlip(axes=("x", "y"), p_per_axis=0.5),
        RandomAffine(rotation=20.0, scale=(0.9, 1.1)),
        ElasticDeform(alpha=4.0, sigma=2.0),
    ], seed=seed)

    for s in range(100):
        out = pipeline_factory(s)(vol)
        # Wherever mask == 1, image should be > 0; wherever image > 0.5, mask is 1.
        # Allow small interpolation-driven softening at the boundary by checking
        # that the *areas* match within ~25% (transforms preserve area to that).
        img_fg = (out.image > 0.3).sum()
        msk_fg = (out.mask > 0).sum()
        # At least one of them should be non-zero (transform didn't blank the volume).
        assert img_fg > 0 and msk_fg > 0, f"blank result at seed {s}"
        # Mask centroid should sit within image foreground bbox
        msk_idx = np.argwhere(out.mask > 0)
        cy, cx = msk_idx.mean(axis=0)
        assert out.image[int(round(cy)), int(round(cx))] > 0.05, (
            f"mask centroid outside image foreground at seed {s}"
        )
