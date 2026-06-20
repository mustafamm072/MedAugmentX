import numpy as np
import pytest

from medaugmentx.core import MedVolume
from medaugmentx.transforms import MetalStreak


@pytest.fixture
def vol2d():
    img = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    mask = (img > 0.5).astype(np.uint8)
    return MedVolume(image=img, mask=mask)


@pytest.fixture
def vol3d():
    img = np.random.default_rng(1).random((6, 48, 48)).astype(np.float32)
    return MedVolume(image=img)


def test_shape_dtype_preserved_3d(vol3d):
    out = MetalStreak(seed=0)(vol3d)
    assert out.image.shape == vol3d.image.shape
    assert out.image.dtype == np.float32


def test_changes_image(vol2d):
    out = MetalStreak(intensity=0.3, seed=0)(vol2d)
    assert not np.allclose(out.image, vol2d.image)


def test_mask_untouched(vol2d):
    out = MetalStreak(seed=0)(vol2d)
    np.testing.assert_array_equal(out.mask, vol2d.mask)


def test_intensity_zero_identity(vol2d):
    out = MetalStreak(intensity=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_same_pattern_across_slices(vol3d):
    out = MetalStreak(intensity=0.3, seed=0)(vol3d)
    diff = out.image - vol3d.image
    # The added streak field is identical on every slice.
    np.testing.assert_allclose(diff[0], diff[1], atol=1e-5)


def test_deterministic(vol2d):
    a = MetalStreak(intensity=(0.1, 0.3), seed=3)(vol2d)
    b = MetalStreak(intensity=(0.1, 0.3), seed=3)(vol2d)
    np.testing.assert_array_equal(a.image, b.image)


def test_p_zero_no_op(vol2d):
    out = MetalStreak(p=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        MetalStreak(intensity=-0.1)
    with pytest.raises(ValueError):
        MetalStreak(falloff=0.0)
    with pytest.raises(ValueError):
        MetalStreak(num_streaks=(5, 2))


def test_to_dict_round_trip():
    t = MetalStreak(intensity=(0.1, 0.3), num_streaks=(6, 12), num_sources=2, falloff=0.5, p=0.4)
    d = t.to_dict()
    assert d["name"] == "MetalStreak"
    t2 = MetalStreak(**d["params"])
    assert t2.intensity_range == t.intensity_range
    assert t2.streak_range == t.streak_range
    assert t2.source_range == t.source_range
