import numpy as np
import pytest

from medaugment.core import MedVolume
from medaugment.transforms import BrightnessContrast


@pytest.fixture
def vol():
    img = np.full((32, 32), 0.5, dtype=np.float32)
    mask = np.ones((32, 32), dtype=np.uint8)
    return MedVolume(image=img, mask=mask)


def test_brightness_contrast_changes_image(vol):
    out = BrightnessContrast(brightness=0.1, contrast=(0.9, 1.1), seed=0)(vol)
    assert not np.allclose(out.image, vol.image)


def test_brightness_contrast_identity(vol):
    out = BrightnessContrast(brightness=0.0, contrast=1.0, seed=0)(vol)
    np.testing.assert_allclose(out.image, vol.image, atol=1e-6)


def test_brightness_contrast_clip_respected(vol):
    out = BrightnessContrast(brightness=1.0, contrast=2.0, clip=(0.0, 1.0), seed=0)(vol)
    assert out.image.min() >= 0.0
    assert out.image.max() <= 1.0


def test_brightness_contrast_mask_untouched(vol):
    out = BrightnessContrast(brightness=0.1, seed=0)(vol)
    np.testing.assert_array_equal(out.mask, vol.mask)


def test_brightness_contrast_deterministic(vol):
    a = BrightnessContrast(brightness=0.05, contrast=(0.9, 1.1), seed=99)(vol)
    b = BrightnessContrast(brightness=0.05, contrast=(0.9, 1.1), seed=99)(vol)
    np.testing.assert_array_equal(a.image, b.image)


def test_brightness_contrast_shape_preserved(vol):
    out = BrightnessContrast(seed=0)(vol)
    assert out.image.shape == vol.image.shape
    assert out.image.dtype == np.float32


def test_brightness_contrast_p_zero_no_op(vol):
    out = BrightnessContrast(brightness=0.5, p=0.0, seed=0)(vol)
    np.testing.assert_array_equal(out.image, vol.image)


def test_brightness_contrast_invalid_contrast_raises():
    with pytest.raises(ValueError):
        BrightnessContrast(contrast=(-0.5, 0.5))


def test_brightness_contrast_to_dict_round_trip():
    t = BrightnessContrast(brightness=0.05, contrast=(0.9, 1.1), p=0.6)
    d = t.to_dict()
    assert d["name"] == "BrightnessContrast"
    t2 = BrightnessContrast(**d["params"])
    assert t2.brightness_range == t.brightness_range
    assert t2.contrast_range == t.contrast_range
    assert t2.p == t.p
