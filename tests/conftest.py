"""Shared fixtures for the MedAugment test suite."""
from __future__ import annotations

import numpy as np
import pytest

from medaugmentx.core import MedVolume


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def vol2d(rng):
    image = rng.random((64, 64), dtype=np.float64).astype(np.float32)
    mask = (image > 0.5).astype(np.uint8)
    return MedVolume(image=image, mask=mask, spacing=(1.0, 1.0), metadata={"modality": "DX"})


@pytest.fixture
def vol3d(rng):
    image = rng.random((16, 32, 32), dtype=np.float64).astype(np.float32)
    mask = (image > 0.7).astype(np.uint8)
    return MedVolume(
        image=image, mask=mask, spacing=(2.5, 0.7, 0.7), metadata={"modality": "MR"}
    )


@pytest.fixture
def dbt_volume(rng):
    """Anisotropic-spacing 3D volume that mimics a small DBT slab."""
    image = rng.random((20, 64, 64), dtype=np.float64).astype(np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[5:15, 20:40, 20:40] = 1
    return MedVolume(
        image=image,
        mask=mask,
        spacing=(1.0, 0.1, 0.1),
        metadata={"modality": "DBT", "vendor": "generic"},
    )
