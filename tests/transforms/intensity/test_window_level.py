import numpy as np
import pytest

from medaugment.core import MedVolume
from medaugment.transforms import WindowLevel


@pytest.fixture
def ct_vol():
    rng = np.random.default_rng(0)
    img = rng.uniform(-1000, 500, (32, 32)).astype(np.float32)
    return MedVolume(image=img, metadata={"window_center": -600, "window_width": 1500})


@pytest.fixture
def plain_vol():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, (32, 32)).astype(np.float32)
    return MedVolume(image=img)


def test_window_level_rescaled_output_range(ct_vol):
    out = WindowLevel(rescale_output=True, seed=0)(ct_vol)
    assert out.image.min() >= 0.0 - 1e-5
    assert out.image.max() <= 1.0 + 1e-5


def test_window_level_no_rescale_clips(ct_vol):
    out = WindowLevel(rescale_output=False, center_shift_frac=0.0, width_scale=1.0, seed=0)(ct_vol)
    assert out.image.min() >= ct_vol.image.min() - 1.0
    assert out.image.max() <= ct_vol.image.max() + 1.0


def test_window_level_uses_metadata(ct_vol):
    t = WindowLevel(center_shift_frac=0.0, width_scale=1.0, rescale_output=True, seed=0)
    out = t(ct_vol)
    assert out.image.shape == ct_vol.image.shape


def test_window_level_fallback_no_metadata(plain_vol):
    out = WindowLevel(seed=1)(plain_vol)
    assert out.image.shape == plain_vol.image.shape


def test_window_level_mask_untouched():
    img = np.linspace(-500, 500, 64 * 64, dtype=np.float32).reshape(64, 64)
    mask = (img > 0).astype(np.uint8)
    vol = MedVolume(image=img, mask=mask)
    out = WindowLevel(seed=0)(vol)
    np.testing.assert_array_equal(out.mask, mask)


def test_window_level_deterministic(ct_vol):
    a = WindowLevel(seed=42)(ct_vol)
    b = WindowLevel(seed=42)(ct_vol)
    np.testing.assert_array_equal(a.image, b.image)


def test_window_level_p_zero_no_op(plain_vol):
    out = WindowLevel(p=0.0, seed=0)(plain_vol)
    np.testing.assert_array_equal(out.image, plain_vol.image)


def test_window_level_invalid_width_scale():
    with pytest.raises(ValueError):
        WindowLevel(width_scale=-0.5)


def test_window_level_to_dict_round_trip():
    t = WindowLevel(center_shift_frac=0.1, width_scale=(0.8, 1.2), p=0.7)
    d = t.to_dict()
    assert d["name"] == "WindowLevel"
    t2 = WindowLevel(**d["params"])
    assert t2.width_scale_range == t.width_scale_range
    assert t2.p == t.p
