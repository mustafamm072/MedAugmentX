import numpy as np
import pytest

from medaugmentx.core import MedVolume
from medaugmentx.transforms import MedianBlur


@pytest.fixture
def vol2d():
    img = np.random.default_rng(0).random((48, 48)).astype(np.float32)
    mask = (img > 0.5).astype(np.uint8)
    return MedVolume(image=img, mask=mask)


@pytest.fixture
def vol3d():
    img = np.random.default_rng(1).random((8, 24, 24)).astype(np.float32)
    return MedVolume(image=img)


def test_shape_dtype_preserved_3d(vol3d):
    out = MedianBlur(ksize=3, seed=0)(vol3d)
    assert out.image.shape == vol3d.image.shape
    assert out.image.dtype == np.float32


def test_removes_salt_and_pepper(vol2d):
    img = vol2d.image.copy()
    img[5, 5] = 100.0  # impulse
    v = MedVolume(image=img)
    out = MedianBlur(ksize=3, seed=0)(v)
    assert out.image[5, 5] < 10.0


def test_mask_untouched(vol2d):
    out = MedianBlur(ksize=3, seed=0)(vol2d)
    np.testing.assert_array_equal(out.mask, vol2d.mask)


def test_ksize_one_identity(vol2d):
    out = MedianBlur(ksize=1, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_deterministic(vol2d):
    a = MedianBlur(ksize=(3, 5), seed=3)(vol2d)
    b = MedianBlur(ksize=(3, 5), seed=3)(vol2d)
    np.testing.assert_array_equal(a.image, b.image)


def test_p_zero_no_op(vol2d):
    out = MedianBlur(ksize=3, p=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_even_ksize_raises():
    with pytest.raises(ValueError):
        MedianBlur(ksize=4)
    with pytest.raises(ValueError):
        MedianBlur(ksize=(2, 4))


def test_to_dict_round_trip():
    t = MedianBlur(ksize=(3, 5), mode="reflect", p=0.5)
    d = t.to_dict()
    assert d["name"] == "MedianBlur"
    t2 = MedianBlur(**d["params"])
    assert t2.ksize_range == t.ksize_range
