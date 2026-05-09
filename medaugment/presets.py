"""Pre-built augmentation pipelines for common medical imaging modalities.

Each factory returns a :class:`~medaugment.core.compose.Compose` pipeline
configured with clinically-motivated defaults.  They are starting points —
tune parameters to your dataset rather than treating them as ground truth.

Available presets::

    from medaugment.presets import mri_pipeline, ct_pipeline, dxr_pipeline, dbt_pipeline

    pipeline = mri_pipeline(seed=42)
    augmented = pipeline(volume)

All pipelines are serialisable via :mod:`medaugment.serialization`.
"""
from __future__ import annotations

from medaugment.core.compose import Compose, OneOf, SomeOf
from medaugment.core.utils import SeedLike

# ---------------------------------------------------------------------------
# MRI
# ---------------------------------------------------------------------------


def mri_pipeline(seed: SeedLike = None) -> Compose:
    """Standard MRI augmentation pipeline.

    Applies a representative mix of spatial deformations, intensity
    perturbations, and MRI-specific artifacts:

    * Random flip (L–R, the only anatomically valid flip for most brain MRI).
    * Random affine — small rotations and scaling.
    * Elastic deformation — simulate tissue deformability.
    * Bias field — RF coil / B0 inhomogeneity.
    * Rician noise — the physically correct MRI noise model.
    * Gamma correction — scanner contrast variation.
    * One of: ghosting artifact *or* k-space line dropout.

    Args:
        seed: Top-level RNG seed for reproducibility.

    Returns:
        A seeded :class:`~medaugment.core.compose.Compose` pipeline.
    """
    from medaugment.transforms.intensity.bias_field import BiasField
    from medaugment.transforms.intensity.contrast import GammaCorrection
    from medaugment.transforms.intensity.noise import RicianNoise
    from medaugment.transforms.modality.mri.ghosting import GhostingArtifact
    from medaugment.transforms.modality.mri.kspace import KSpaceDropout
    from medaugment.transforms.spatial.affine import RandomAffine
    from medaugment.transforms.spatial.elastic import ElasticDeform
    from medaugment.transforms.spatial.flip import RandomFlip

    return Compose(
        [
            RandomFlip(axes=("x",), p_per_axis=0.5, p=0.5),
            RandomAffine(rotation=10.0, scale=(0.9, 1.1), translation=(-0.05, 0.05), p=0.7),
            ElasticDeform(alpha=30.0, sigma=4.0, p=0.5),
            BiasField(alpha=0.3, p=0.7),
            RicianNoise(std=(0.005, 0.02), p=0.5),
            GammaCorrection(gamma=(0.85, 1.15), p=0.5),
            OneOf(
                [GhostingArtifact(ghost_intensity=(0.05, 0.12), p=1.0),
                 KSpaceDropout(dropout_fraction=(0.01, 0.04), p=1.0)],
                p=0.3,
            ),
        ],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# CT
# ---------------------------------------------------------------------------


def ct_pipeline(seed: SeedLike = None) -> Compose:
    """Standard CT augmentation pipeline.

    Combines spatial augmentations with CT-specific intensity transforms:

    * Random flip along L–R.
    * Random affine — small rotations and scaling.
    * Elastic deformation.
    * Window / level variation — simulates protocol differences across sites.
    * Gaussian noise — additive CT quantum noise.
    * Gamma correction — contrast variation.
    * Beam hardening (cupping artifact) — applied occasionally.

    Args:
        seed: Top-level RNG seed for reproducibility.

    Returns:
        A seeded :class:`~medaugment.core.compose.Compose` pipeline.
    """
    from medaugment.transforms.intensity.contrast import GammaCorrection
    from medaugment.transforms.intensity.noise import GaussianNoise
    from medaugment.transforms.intensity.window_level import WindowLevel
    from medaugment.transforms.modality.ct.beam_hardening import BeamHardening
    from medaugment.transforms.spatial.affine import RandomAffine
    from medaugment.transforms.spatial.elastic import ElasticDeform
    from medaugment.transforms.spatial.flip import RandomFlip

    return Compose(
        [
            RandomFlip(axes=("x",), p_per_axis=0.5, p=0.5),
            RandomAffine(rotation=8.0, scale=(0.9, 1.1), translation=(-0.03, 0.03), p=0.7),
            ElasticDeform(alpha=20.0, sigma=4.0, p=0.4),
            WindowLevel(center_shift_frac=0.05, width_scale=(0.85, 1.15), p=0.6),
            GaussianNoise(std=(5.0, 20.0), p=0.4),
            GammaCorrection(gamma=(0.9, 1.1), p=0.4),
            BeamHardening(alpha=(0.02, 0.07), p=0.3),
        ],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Digital X-Ray (DXR / Chest X-Ray)
# ---------------------------------------------------------------------------


def dxr_pipeline(seed: SeedLike = None) -> Compose:
    """Augmentation pipeline for digital X-ray (chest X-ray / DXR).

    Applies 2-D-safe transforms only:

    * Horizontal flip (L–R chest flip is anatomically valid in practice).
    * Slight affine — small rotation (patient positioning variation).
    * Gaussian blur — simulates detector resolution variation.
    * Brightness/contrast — exposure variation between scanners.
    * Gamma correction — film/detector response variation.
    * Simulate low resolution — cross-site resolution augmentation.

    Args:
        seed: Top-level RNG seed for reproducibility.

    Returns:
        A seeded :class:`~medaugment.core.compose.Compose` pipeline.
    """
    from medaugment.transforms.intensity.blur import GaussianBlur, SimulateLowResolution
    from medaugment.transforms.intensity.brightness_contrast import BrightnessContrast
    from medaugment.transforms.intensity.contrast import GammaCorrection
    from medaugment.transforms.spatial.affine import RandomAffine
    from medaugment.transforms.spatial.flip import RandomFlip

    return Compose(
        [
            RandomFlip(axes=("x",), p_per_axis=0.5, p=0.5),
            RandomAffine(
                rotation=5.0,
                scale=(0.95, 1.05),
                axes_enabled=("x", "y"),
                p=0.6,
            ),
            GaussianBlur(sigma=(0.3, 1.2), p=0.4),
            BrightnessContrast(brightness=0.05, contrast=(0.9, 1.1), p=0.5),
            GammaCorrection(gamma=(0.8, 1.2), p=0.5),
            SimulateLowResolution(zoom_range=(0.6, 0.95), p=0.3),
        ],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Digital Breast Tomosynthesis (DBT)
# ---------------------------------------------------------------------------


def dbt_pipeline(seed: SeedLike = None) -> Compose:
    """Augmentation pipeline for Digital Breast Tomosynthesis (DBT).

    DBT volumes are strongly anisotropic (thin slices, wide in-plane FOV).
    This pipeline combines the Phase-1 DBT-specific transforms with the
    new Phase-2 intensity transforms:

    * Random flip — horizontal axis only (breast anatomy).
    * Anisotropic affine — in-plane rotation only (``axes_enabled=("x","y")``).
    * Anisotropic elastic deformation — larger XY alpha, small Z.
    * Slab shift — reconstruction centre variation.
    * Limited-angle blur — arc-angle-dependent Z blur.
    * Slice dropout — robustness to missing reconstruction slices.
    * Bias field — detector gain non-uniformity.
    * Gamma correction — scatter / beam hardening contrast variation.

    Args:
        seed: Top-level RNG seed for reproducibility.

    Returns:
        A seeded :class:`~medaugment.core.compose.Compose` pipeline.
    """
    from medaugment.transforms.intensity.bias_field import BiasField
    from medaugment.transforms.intensity.contrast import GammaCorrection
    from medaugment.transforms.modality.tomosynthesis.blur import LimitedAngleBlur
    from medaugment.transforms.modality.tomosynthesis.dropout import SliceDropout
    from medaugment.transforms.modality.tomosynthesis.elastic import AnisotropicElastic
    from medaugment.transforms.modality.tomosynthesis.slab import SlabShift
    from medaugment.transforms.spatial.affine import RandomAffine
    from medaugment.transforms.spatial.flip import RandomFlip

    return Compose(
        [
            RandomFlip(axes=("x",), p_per_axis=0.5, p=0.5),
            RandomAffine(
                rotation=5.0,
                scale=(0.95, 1.05),
                axes_enabled=("x", "y"),
                p=0.7,
            ),
            AnisotropicElastic(
                alpha=(80.0, 80.0, 6.0),
                sigma=(8.0, 8.0, 2.0),
                p=0.5,
            ),
            SlabShift(max_shift=2, p=0.5),
            LimitedAngleBlur(arc_degrees=(15.0, 25.0), p=0.6),
            SliceDropout(num_slices=(1, 2), p=0.3),
            BiasField(alpha=0.2, coarse_shape=3, p=0.5),
            GammaCorrection(gamma=(0.85, 1.15), p=0.5),
        ],
        seed=seed,
    )


__all__ = ["mri_pipeline", "ct_pipeline", "dxr_pipeline", "dbt_pipeline"]
