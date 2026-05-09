"""Tests for CT-specific transforms: BeamHardening."""
import numpy as np
import pytest

from medaugment.core import MedVolume
from medaugment.transforms import BeamHardening


@pytest.fixture
def ct_vol2d():
    rng = np.random.default_rng(0)
    img = rng.uniform(-200, 300, (64, 64)).astype(np.float32)
    return MedVolume(image=img)


@pytest.fixture
def ct_vol3d():
    rng = np.random.default_rng(1)
    img = rng.uniform(-200, 300, (8, 64, 64)).astype(np.float32)
    mask = (img > 0).astype(np.uint8)
    return MedVolume(image=img, mask=mask)


def test_beam_hardening_changes_image_2d(ct_vol2d):
    out = BeamHardening(alpha=0.05, seed=0)(ct_vol2d)
    assert not np.allclose(out.image, ct_vol2d.image)


def test_beam_hardening_changes_image_3d(ct_vol3d):
    out = BeamHardening(alpha=0.05, seed=0)(ct_vol3d)
    assert not np.allclose(out.image, ct_vol3d.image)
    assert out.image.shape == ct_vol3d.image.shape


def test_beam_hardening_centre_darker_than_edge(ct_vol2d):
    out = BeamHardening(alpha=0.1, power=2.0, seed=0)(ct_vol2d)
    centre_diff = ct_vol2d.image[32, 32] - out.image[32, 32]
    edge_diff = ct_vol2d.image[0, 0] - out.image[0, 0]
    # Centre should be darkened more than corner
    assert centre_diff > edge_diff


def test_beam_hardening_alpha_zero_is_near_identity():
    img = np.ones((32, 32), dtype=np.float32)
    vol = MedVolume(image=img)
    out = BeamHardening(alpha=0.0, seed=0)(vol)
    np.testing.assert_allclose(out.image, vol.image, atol=1e-6)


def test_beam_hardening_mask_untouched(ct_vol3d):
    out = BeamHardening(alpha=0.05, seed=0)(ct_vol3d)
    np.testing.assert_array_equal(out.mask, ct_vol3d.mask)


def test_beam_hardening_deterministic(ct_vol2d):
    a = BeamHardening(alpha=(0.02, 0.08), seed=13)(ct_vol2d)
    b = BeamHardening(alpha=(0.02, 0.08), seed=13)(ct_vol2d)
    np.testing.assert_array_equal(a.image, b.image)


def test_beam_hardening_p_zero_no_op(ct_vol2d):
    out = BeamHardening(alpha=0.1, p=0.0, seed=0)(ct_vol2d)
    np.testing.assert_array_equal(out.image, ct_vol2d.image)


def test_beam_hardening_constant_image_no_op():
    img = np.full((32, 32), 5.0, dtype=np.float32)
    vol = MedVolume(image=img)
    out = BeamHardening(alpha=0.05, seed=0)(vol)
    np.testing.assert_array_equal(out.image, img)


def test_beam_hardening_invalid_alpha_raises():
    with pytest.raises(ValueError):
        BeamHardening(alpha=-0.05)


def test_beam_hardening_invalid_power_raises():
    with pytest.raises(ValueError):
        BeamHardening(power=0.0)


def test_beam_hardening_to_dict_round_trip():
    t = BeamHardening(alpha=(0.02, 0.08), power=2.0, p=0.4)
    d = t.to_dict()
    assert d["name"] == "BeamHardening"
    t2 = BeamHardening(**d["params"])
    assert t2.alpha_range == t.alpha_range
    assert t2.power == t.power
    assert t2.p == t.p
