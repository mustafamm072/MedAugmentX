import numpy as np
import pytest

from medaugmentx.core import MedVolume
from medaugmentx.transforms import CompressionVariation, ReconStreak


@pytest.fixture
def dbt_vol():
    img = np.random.default_rng(0).random((12, 48, 48)).astype(np.float32)
    mask = np.zeros((12, 48, 48), dtype=np.uint8)
    mask[3:9, 12:36, 12:36] = 1
    return MedVolume(image=img, mask=mask, spacing=(1.0, 0.1, 0.1))


@pytest.fixture
def vol2d():
    return MedVolume(image=np.random.default_rng(0).random((32, 32)).astype(np.float32))


# ---- CompressionVariation ----

def test_compression_shape_preserved(dbt_vol):
    out = CompressionVariation(scale=0.85, seed=0)(dbt_vol)
    assert out.image.shape == dbt_vol.image.shape
    assert out.mask.shape == dbt_vol.mask.shape
    assert out.image.dtype == np.float32


def test_compression_mask_consistent(dbt_vol):
    out = CompressionVariation(scale=0.85, seed=0)(dbt_vol)
    assert out.mask.dtype == dbt_vol.mask.dtype
    assert set(np.unique(out.mask).tolist()) <= {0, 1}


def test_compression_changes_image(dbt_vol):
    out = CompressionVariation(scale=0.8, seed=0)(dbt_vol)
    assert not np.allclose(out.image, dbt_vol.image)


def test_compression_scale_one_identity(dbt_vol):
    out = CompressionVariation(scale=1.0, seed=0)(dbt_vol)
    np.testing.assert_array_equal(out.image, dbt_vol.image)


def test_compression_rejects_2d(vol2d):
    with pytest.raises(ValueError):
        CompressionVariation(seed=0)(vol2d)


def test_compression_deterministic(dbt_vol):
    a = CompressionVariation(scale=(0.85, 1.15), seed=4)(dbt_vol)
    b = CompressionVariation(scale=(0.85, 1.15), seed=4)(dbt_vol)
    np.testing.assert_array_equal(a.image, b.image)


def test_compression_invalid_params_raise():
    with pytest.raises(ValueError):
        CompressionVariation(scale=0.0)
    with pytest.raises(ValueError):
        CompressionVariation(axis="z")


def test_compression_to_dict_round_trip():
    t = CompressionVariation(scale=(0.85, 1.15), axis="y", order=1, p=0.5)
    t2 = CompressionVariation(**t.to_dict()["params"])
    assert t2.scale_range == t.scale_range
    assert t2.axis == t.axis


# ---- ReconStreak ----

def test_recon_shape_dtype(dbt_vol):
    out = ReconStreak(seed=0)(dbt_vol)
    assert out.image.shape == dbt_vol.image.shape
    assert out.image.dtype == np.float32


def test_recon_changes_image(dbt_vol):
    out = ReconStreak(amplitude=0.2, seed=0)(dbt_vol)
    assert not np.allclose(out.image, dbt_vol.image)


def test_recon_mask_untouched(dbt_vol):
    out = ReconStreak(seed=0)(dbt_vol)
    np.testing.assert_array_equal(out.mask, dbt_vol.mask)


def test_recon_amplitude_zero_identity(dbt_vol):
    out = ReconStreak(amplitude=0.0, seed=0)(dbt_vol)
    np.testing.assert_array_equal(out.image, dbt_vol.image)


def test_recon_rejects_2d(vol2d):
    with pytest.raises(ValueError):
        ReconStreak(seed=0)(vol2d)


def test_recon_deterministic(dbt_vol):
    a = ReconStreak(amplitude=(0.05, 0.2), seed=6)(dbt_vol)
    b = ReconStreak(amplitude=(0.05, 0.2), seed=6)(dbt_vol)
    np.testing.assert_array_equal(a.image, b.image)


def test_recon_invalid_params_raise():
    with pytest.raises(ValueError):
        ReconStreak(amplitude=-0.1)
    with pytest.raises(ValueError):
        ReconStreak(decay=1.5)
    with pytest.raises(ValueError):
        ReconStreak(num_planes=(3, 1))


def test_recon_to_dict_round_trip():
    t = ReconStreak(amplitude=(0.05, 0.2), num_planes=(1, 3), displacement=1.5, decay=0.6, axis="x", p=0.5)
    d = t.to_dict()
    assert d["name"] == "ReconStreak"
    t2 = ReconStreak(**d["params"])
    assert t2.amplitude_range == t.amplitude_range
    assert t2.plane_range == t.plane_range
    assert t2.decay == t.decay
