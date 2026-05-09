import numpy as np
import pytest

from medaugment.core import MedVolume
from medaugment.transforms import GaussianBlur, SimulateLowResolution


@pytest.fixture
def vol2d():
    rng = np.random.default_rng(0)
    img = rng.random((64, 64)).astype(np.float32)
    mask = (img > 0.5).astype(np.uint8)
    return MedVolume(image=img, mask=mask)


@pytest.fixture
def vol3d():
    rng = np.random.default_rng(1)
    img = rng.random((16, 32, 32)).astype(np.float32)
    mask = (img > 0.7).astype(np.uint8)
    return MedVolume(image=img, mask=mask)


# ---- GaussianBlur ----

def test_gaussian_blur_2d_smooths_image(vol2d):
    out = GaussianBlur(sigma=2.0, seed=0)(vol2d)
    # Blur reduces std
    assert out.image.std() < vol2d.image.std()


def test_gaussian_blur_3d_shape_preserved(vol3d):
    out = GaussianBlur(sigma=1.0, seed=0)(vol3d)
    assert out.image.shape == vol3d.image.shape


def test_gaussian_blur_zero_sigma_identity(vol2d):
    out = GaussianBlur(sigma=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_gaussian_blur_mask_untouched(vol2d):
    out = GaussianBlur(sigma=1.5, seed=0)(vol2d)
    np.testing.assert_array_equal(out.mask, vol2d.mask)


def test_gaussian_blur_deterministic(vol2d):
    a = GaussianBlur(sigma=(0.5, 2.0), seed=42)(vol2d)
    b = GaussianBlur(sigma=(0.5, 2.0), seed=42)(vol2d)
    np.testing.assert_array_equal(a.image, b.image)


def test_gaussian_blur_p_zero_no_op(vol2d):
    out = GaussianBlur(sigma=2.0, p=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_gaussian_blur_invalid_sigma_raises():
    with pytest.raises(ValueError):
        GaussianBlur(sigma=-1.0)


def test_gaussian_blur_to_dict_round_trip():
    t = GaussianBlur(sigma=(0.5, 1.5), order=0, mode="reflect", p=0.7)
    d = t.to_dict()
    assert d["name"] == "GaussianBlur"
    t2 = GaussianBlur(**d["params"])
    assert t2.sigma_range == t.sigma_range
    assert t2.p == t.p


# ---- SimulateLowResolution ----

def test_low_res_shape_preserved_2d(vol2d):
    out = SimulateLowResolution(zoom_range=(0.5, 0.9), seed=0)(vol2d)
    assert out.image.shape == vol2d.image.shape


def test_low_res_shape_preserved_3d(vol3d):
    out = SimulateLowResolution(zoom_range=(0.5, 0.9), seed=0)(vol3d)
    assert out.image.shape == vol3d.image.shape


def test_low_res_mask_untouched(vol2d):
    out = SimulateLowResolution(zoom_range=(0.5, 0.9), seed=0)(vol2d)
    np.testing.assert_array_equal(out.mask, vol2d.mask)


def test_low_res_deterministic(vol2d):
    a = SimulateLowResolution(zoom_range=(0.5, 0.9), seed=10)(vol2d)
    b = SimulateLowResolution(zoom_range=(0.5, 0.9), seed=10)(vol2d)
    np.testing.assert_array_equal(a.image, b.image)


def test_low_res_changes_image(vol2d):
    out = SimulateLowResolution(zoom_range=(0.4, 0.6), seed=0)(vol2d)
    assert not np.allclose(out.image, vol2d.image)


def test_low_res_p_zero_no_op(vol2d):
    out = SimulateLowResolution(zoom_range=(0.5, 0.8), p=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_low_res_invalid_zoom_range_raises():
    with pytest.raises(ValueError):
        SimulateLowResolution(zoom_range=(0.0, 0.5))
    with pytest.raises(ValueError):
        SimulateLowResolution(zoom_range=(0.5, 1.5))


def test_low_res_to_dict_round_trip():
    t = SimulateLowResolution(zoom_range=(0.5, 0.9), per_axis=True, p=0.5)
    d = t.to_dict()
    assert d["name"] == "SimulateLowResolution"
    t2 = SimulateLowResolution(**d["params"])
    assert t2.zoom_range == t.zoom_range
    assert t2.per_axis == t.per_axis
