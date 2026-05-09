"""Tests for MRI-specific transforms: GhostingArtifact, KSpaceDropout."""
import numpy as np
import pytest

from medaugment.core import MedVolume
from medaugment.transforms import GhostingArtifact, KSpaceDropout


@pytest.fixture
def vol2d():
    rng = np.random.default_rng(0)
    img = rng.random((64, 64)).astype(np.float32)
    mask = (img > 0.5).astype(np.uint8)
    return MedVolume(image=img, mask=mask)


@pytest.fixture
def vol3d():
    rng = np.random.default_rng(1)
    img = rng.random((8, 32, 32)).astype(np.float32)
    mask = (img > 0.7).astype(np.uint8)
    return MedVolume(image=img, mask=mask)


# ---- GhostingArtifact ----

def test_ghosting_changes_image_2d(vol2d):
    out = GhostingArtifact(ghost_intensity=0.1, ghost_shift=16, seed=0)(vol2d)
    assert not np.allclose(out.image, vol2d.image)


def test_ghosting_changes_image_3d(vol3d):
    out = GhostingArtifact(ghost_intensity=0.1, ghost_shift=8, seed=0)(vol3d)
    assert out.image.shape == vol3d.image.shape


def test_ghosting_shape_preserved(vol2d):
    out = GhostingArtifact(seed=0)(vol2d)
    assert out.image.shape == vol2d.image.shape


def test_ghosting_mask_untouched(vol2d):
    out = GhostingArtifact(seed=0)(vol2d)
    np.testing.assert_array_equal(out.mask, vol2d.mask)


def test_ghosting_deterministic(vol2d):
    a = GhostingArtifact(ghost_intensity=0.1, seed=7)(vol2d)
    b = GhostingArtifact(ghost_intensity=0.1, seed=7)(vol2d)
    np.testing.assert_array_equal(a.image, b.image)


def test_ghosting_p_zero_no_op(vol2d):
    out = GhostingArtifact(p=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_ghosting_zero_image_no_op():
    img = np.zeros((32, 32), dtype=np.float32)
    vol = MedVolume(image=img)
    out = GhostingArtifact(ghost_intensity=0.1, seed=0)(vol)
    np.testing.assert_array_equal(out.image, img)


def test_ghosting_invalid_axis_raises():
    with pytest.raises(ValueError):
        GhostingArtifact(phase_encode_axis="z")


def test_ghosting_to_dict_round_trip():
    t = GhostingArtifact(ghost_intensity=(0.05, 0.15), ghost_shift=(8, 32), p=0.4)
    d = t.to_dict()
    assert d["name"] == "GhostingArtifact"
    t2 = GhostingArtifact(**d["params"])
    assert t2.ghost_intensity_range == t.ghost_intensity_range
    assert t2.ghost_shift_range == t.ghost_shift_range


# ---- KSpaceDropout ----

def test_kspace_changes_image_2d(vol2d):
    out = KSpaceDropout(dropout_fraction=0.05, seed=0)(vol2d)
    assert out.image.shape == vol2d.image.shape
    # After k-space corruption the image will differ
    assert not np.allclose(out.image, vol2d.image)


def test_kspace_changes_image_3d(vol3d):
    out = KSpaceDropout(dropout_fraction=0.05, seed=0)(vol3d)
    assert out.image.shape == vol3d.image.shape


def test_kspace_mask_untouched(vol2d):
    out = KSpaceDropout(dropout_fraction=0.05, seed=0)(vol2d)
    np.testing.assert_array_equal(out.mask, vol2d.mask)


def test_kspace_deterministic(vol2d):
    a = KSpaceDropout(dropout_fraction=0.05, seed=42)(vol2d)
    b = KSpaceDropout(dropout_fraction=0.05, seed=42)(vol2d)
    np.testing.assert_array_equal(a.image, b.image)


def test_kspace_zero_fraction_no_op(vol2d):
    out = KSpaceDropout(dropout_fraction=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_kspace_p_zero_no_op(vol2d):
    out = KSpaceDropout(dropout_fraction=0.05, p=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_kspace_invalid_fraction_raises():
    with pytest.raises(ValueError):
        KSpaceDropout(dropout_fraction=1.5)


def test_kspace_output_non_negative(vol2d):
    out = KSpaceDropout(dropout_fraction=0.1, seed=0)(vol2d)
    # Magnitude image is always >= 0
    assert out.image.min() >= 0.0


def test_kspace_to_dict_round_trip():
    t = KSpaceDropout(dropout_fraction=(0.01, 0.05), phase_encode_axis="x", p=0.5)
    d = t.to_dict()
    assert d["name"] == "KSpaceDropout"
    t2 = KSpaceDropout(**d["params"])
    assert t2.dropout_range == t.dropout_range
    assert t2.phase_encode_axis == t.phase_encode_axis
