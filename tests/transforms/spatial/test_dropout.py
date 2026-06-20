import numpy as np
import pytest

from medaugmentx.core import MedVolume
from medaugmentx.transforms import CoarseDropout


@pytest.fixture
def vol2d():
    img = np.ones((50, 50), dtype=np.float32)
    mask = np.ones((50, 50), dtype=np.uint8)
    return MedVolume(image=img, mask=mask)


@pytest.fixture
def vol3d():
    img = np.ones((10, 40, 40), dtype=np.float32)
    mask = np.ones((10, 40, 40), dtype=np.uint8)
    return MedVolume(image=img, mask=mask)


def test_drops_pixels_2d(vol2d):
    out = CoarseDropout(num_holes=3, hole_size=(0.1, 0.1), fill_value=0.0, seed=0)(vol2d)
    assert (out.image == 0).sum() == 3 * 25  # 3 holes of 5x5


def test_shape_preserved_3d(vol3d):
    out = CoarseDropout(num_holes=(1, 4), seed=0)(vol3d)
    assert out.image.shape == vol3d.image.shape
    assert out.image.dtype == np.float32


def test_mask_untouched_by_default(vol2d):
    out = CoarseDropout(num_holes=3, seed=0)(vol2d)
    np.testing.assert_array_equal(out.mask, vol2d.mask)


def test_fill_mask_blanks_mask(vol2d):
    out = CoarseDropout(num_holes=3, hole_size=(0.1, 0.1), fill_mask=True, seed=0)(vol2d)
    assert (out.mask == 0).any()


def test_num_holes_zero_is_noop(vol2d):
    out = CoarseDropout(num_holes=0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_deterministic(vol3d):
    a = CoarseDropout(num_holes=(2, 5), seed=7)(vol3d)
    b = CoarseDropout(num_holes=(2, 5), seed=7)(vol3d)
    np.testing.assert_array_equal(a.image, b.image)


def test_p_zero_no_op(vol2d):
    out = CoarseDropout(num_holes=5, p=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_invalid_hole_size_raises():
    with pytest.raises(ValueError):
        CoarseDropout(hole_size=(0.0, 0.5))
    with pytest.raises(ValueError):
        CoarseDropout(hole_size=(0.2, 1.5))


def test_invalid_num_holes_raises():
    with pytest.raises(ValueError):
        CoarseDropout(num_holes=(5, 2))


def test_to_dict_round_trip():
    t = CoarseDropout(num_holes=(1, 4), hole_size=(0.05, 0.2), fill_value=0.0, fill_mask=True, p=0.5)
    d = t.to_dict()
    assert d["name"] == "CoarseDropout"
    t2 = CoarseDropout(**d["params"])
    assert t2.num_range == t.num_range
    assert t2.hole_size == t.hole_size
    assert t2.fill_mask == t.fill_mask
