"""Synthetic anatomical phantoms for demos, tutorials, and tests.

Every phantom here is generated from analytic shapes and a seeded RNG. Nothing
in this module reads patient data, downloads a dataset, or contacts a network
service, so the results are safe to publish in documentation and reproducible
across machines.

These are illustrations, not simulations. Intensity values are chosen to sit in
a plausible range for their modality (Hounsfield units for CT, arbitrary units
elsewhere) so that window/level and intensity transforms behave sensibly, but
no phantom is a validated physical model of anatomy or scanner physics. Do not
use them to draw conclusions about clinical performance.

    >>> from medaugmentx.phantoms import ct_phantom
    >>> from medaugmentx.transforms import BeamHardening
    >>> volume = ct_phantom()
    >>> augmented = BeamHardening(alpha=0.10, seed=7)(volume)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from medaugmentx.core import MedVolume

__all__ = [
    "mri_phantom",
    "ct_phantom",
    "xray_phantom",
    "dbt_phantom",
    "DEFAULT_SIZE",
    "DEFAULT_SEED",
]

#: Side length in voxels of the 2D phantoms.
DEFAULT_SIZE = 256

#: Seed used for the randomly placed DBT lesions.
DEFAULT_SEED = 2026


def coordinates(size: int = DEFAULT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Return `(y, x)` coordinate grids spanning `[-1, 1]` in both axes."""
    axis = np.linspace(-1, 1, size)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    return y, x


def ellipse(y: np.ndarray, x: np.ndarray, cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    """Boolean mask of the axis-aligned ellipse centred at `(cy, cx)`."""
    return ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 < 1


def mri_phantom(size: int = DEFAULT_SIZE) -> MedVolume:
    """A 2D axial brain-like slice with grey/white contrast and ventricles.

    Intensities are arbitrary units in roughly `[0, 1]`, which is the range
    `BiasField` and `RicianNoise` are tuned for.
    """
    y, x = coordinates(size)
    head = ellipse(y, x, 0, 0, 0.90, 0.73)
    brain = ellipse(y, x, 0, 0, 0.80, 0.65)
    inner = ellipse(y, x, 0, 0, 0.69, 0.54)
    image = 0.12 * head + 0.43 * brain + 0.18 * inner
    folds = np.sin(35 * x + 4 * np.cos(13 * y)) * np.sin(28 * y - 3 * x)
    image += 0.08 * folds * brain
    image[ellipse(y, x, -0.02, -0.12, 0.24, 0.07)] = 0.12
    image[ellipse(y, x, -0.02, 0.12, 0.24, 0.07)] = 0.12
    image[(np.abs(x) < 0.012) & brain] *= 0.65
    return MedVolume(gaussian_filter(image, 0.7).astype(np.float32), metadata={"modality": "MR"})


def ct_phantom(size: int = DEFAULT_SIZE) -> MedVolume:
    """A 2D axial abdomen-like slice in approximate Hounsfield units.

    Air sits near -1000 HU, soft tissue near 45 HU, and a dense implant near
    800 HU, so window/level and beam-hardening transforms see realistic spans.
    """
    y, x = coordinates(size)
    body = ellipse(y, x, 0, 0, 0.77, 0.91)
    image = np.full(y.shape, -1000.0)
    image[body] = -100.0
    image[ellipse(y, x, 0, 0, 0.67, 0.80)] = 45.0
    image[ellipse(y, x, -0.14, -0.35, 0.35, 0.29)] = 85.0
    image[ellipse(y, x, -0.07, 0.42, 0.25, 0.19)] = 65.0
    for cx in (-0.29, 0.29):
        image[ellipse(y, x, 0.22, cx, 0.16, 0.11)] = 110.0
    image[ellipse(y, x, 0.40, 0, 0.15, 0.15)] = 800.0
    image[ellipse(y, x, 0.40, 0, 0.09, 0.09)] = 150.0
    image[ellipse(y, x, 0.12, 0.02, 0.07, 0.07)] = 170.0
    return MedVolume(gaussian_filter(image, 0.65).astype(np.float32), metadata={"modality": "CT"})


def xray_phantom(size: int = DEFAULT_SIZE) -> MedVolume:
    """A 2D chest-radiograph-like projection with ribs, lungs, and mediastinum.

    Intensities are arbitrary units in roughly `[0, 1]`, matching what
    `ScatterSimulation` and `GridArtifact` expect.
    """
    y, x = coordinates(size)
    body = ellipse(y, x, 0, 0, 0.94, 0.79)
    image = 0.05 + 0.42 * body.astype(float)
    lungs = ellipse(y, x, -0.08, -0.31, 0.68, 0.25) | ellipse(y, x, -0.08, 0.31, 0.68, 0.25)
    image[lungs] = 0.15
    image += 0.20 * np.exp(-((x / 0.085) ** 2)) * body
    image += 0.18 * ellipse(y, x, 0.24, 0.10, 0.32, 0.27)
    for height in np.linspace(-0.67, 0.52, 8):
        arc = height + 0.38 * x**2
        image += 0.14 * np.exp(-(((y - arc) / 0.018) ** 2)) * body
    image += 0.035 * np.sin(21 * x + 9 * y) * lungs
    return MedVolume(gaussian_filter(image, 0.7).astype(np.float32), metadata={"modality": "DX"})


def dbt_phantom(seed: int = DEFAULT_SEED) -> MedVolume:
    """A true 3D tomosynthesis-like volume with scattered lesion blobs.

    Shape is `(64, 128, 256)` with anisotropic spacing `(1.0, 0.5, 0.25)` — the
    thick-slice geometry that the DBT transforms are written for. `seed`
    controls lesion placement only; the background tissue is deterministic.
    """
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, 64), np.linspace(-1, 1, 128), np.linspace(-1, 1, 256), indexing="ij"
    )
    tissue = (z / 0.80) ** 2 + (y / 1.15) ** 2 + ((x + 0.33) / 1.15) ** 2 < 1
    image = tissue * (0.16 + 0.035 * np.sin(15 * x + 9 * z) * np.cos(8 * y))
    rng = np.random.default_rng(seed)
    for _ in range(24):
        cz, cx = rng.uniform(-0.6, 0.6), rng.uniform(-0.9, 0.65)
        rz, rx = rng.uniform(0.025, 0.09), rng.uniform(0.025, 0.13)
        blob = np.exp(-(((z - cz) / rz) ** 2) - ((x - cx) / rx) ** 2 - (y / 0.35) ** 2)
        image += rng.uniform(0.2, 0.6) * blob * tissue
    return MedVolume(
        image.astype(np.float32), spacing=(1.0, 0.5, 0.25), metadata={"modality": "DBT"}
    )
