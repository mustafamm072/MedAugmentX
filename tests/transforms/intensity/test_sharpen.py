import numpy as np
import pytest

from medaugmentx.core import MedVolume
from medaugmentx.transforms import Sharpen


@pytest.fixture
def vol2d():
    img = np.random.default_rng(0).random((48, 48)).astype(np.float32)
    mask = (img > 0.5).astype(np.uint8)
    return MedVolume(image=img, mask=mask)


@pytest.fixture
def vol3d():
    img = np.random.default_rng(1).random((8, 24, 24)).astype(np.float32)
    return MedVolume(image=img)


def test_shape_dtype_preserved(vol3d):
    out = Sharpen(seed=0)(vol3d)
    assert out.image.shape == vol3d.image.shape
    assert out.image.dtype == np.float32


def test_increases_high_frequency_energy(vol2d):
    out = Sharpen(alpha=1.0, sigma=1.0, seed=0)(vol2d)
    # Sharpening boosts variance of the high-pass content.
    assert out.image.std() > vol2d.image.std()


def test_mask_untouched(vol2d):
    out = Sharpen(seed=0)(vol2d)
    np.testing.assert_array_equal(out.mask, vol2d.mask)


def test_alpha_zero_identity(vol2d):
    out = Sharpen(alpha=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_clip_bounds_output(vol2d):
    out = Sharpen(alpha=2.0, clip=(0.0, 1.0), seed=0)(vol2d)
    assert out.image.min() >= 0.0 and out.image.max() <= 1.0


def test_deterministic(vol2d):
    a = Sharpen(alpha=(0.2, 0.8), seed=5)(vol2d)
    b = Sharpen(alpha=(0.2, 0.8), seed=5)(vol2d)
    np.testing.assert_array_equal(a.image, b.image)


def test_p_zero_no_op(vol2d):
    out = Sharpen(alpha=1.0, p=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        Sharpen(alpha=-0.5)
    with pytest.raises(ValueError):
        Sharpen(sigma=0.0)


def test_to_dict_round_trip():
    t = Sharpen(alpha=(0.2, 0.8), sigma=(0.7, 1.5), clip=(0.0, 1.0), p=0.6)
    d = t.to_dict()
    assert d["name"] == "Sharpen"
    t2 = Sharpen(**d["params"])
    assert t2.alpha_range == t.alpha_range
    assert t2.sigma_range == t.sigma_range
    assert t2.clip == t.clip
