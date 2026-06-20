import numpy as np
import pytest

from medaugmentx.core import MedVolume
from medaugmentx.transforms import CLAHEContrast


@pytest.fixture
def vol2d():
    img = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    mask = (img > 0.5).astype(np.uint8)
    return MedVolume(image=img, mask=mask)


@pytest.fixture
def vol3d():
    img = np.random.default_rng(1).random((6, 48, 48)).astype(np.float32)
    return MedVolume(image=img)


def test_shape_dtype_preserved_2d(vol2d):
    out = CLAHEContrast(seed=0)(vol2d)
    assert out.image.shape == vol2d.image.shape
    assert out.image.dtype == np.float32


def test_shape_preserved_3d(vol3d):
    out = CLAHEContrast(grid=(4, 4), seed=0)(vol3d)
    assert out.image.shape == vol3d.image.shape


def test_output_within_original_range(vol2d):
    out = CLAHEContrast(clip_limit=3.0, seed=0)(vol2d)
    assert out.image.min() >= vol2d.image.min() - 1e-4
    assert out.image.max() <= vol2d.image.max() + 1e-4


def test_changes_image(vol2d):
    out = CLAHEContrast(clip_limit=3.0, seed=0)(vol2d)
    assert not np.allclose(out.image, vol2d.image)


def test_output_finite(vol2d):
    out = CLAHEContrast(seed=0)(vol2d)
    assert np.isfinite(out.image).all()


def test_constant_image_identity():
    v = MedVolume(image=np.full((32, 32), 0.5, dtype=np.float32))
    out = CLAHEContrast(seed=0)(v)
    np.testing.assert_array_equal(out.image, v.image)


def test_mask_untouched(vol2d):
    out = CLAHEContrast(seed=0)(vol2d)
    np.testing.assert_array_equal(out.mask, vol2d.mask)


def test_deterministic(vol2d):
    a = CLAHEContrast(clip_limit=(1.0, 3.0), seed=11)(vol2d)
    b = CLAHEContrast(clip_limit=(1.0, 3.0), seed=11)(vol2d)
    np.testing.assert_array_equal(a.image, b.image)


def test_p_zero_no_op(vol2d):
    out = CLAHEContrast(p=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        CLAHEContrast(clip_limit=0.0)
    with pytest.raises(ValueError):
        CLAHEContrast(grid=(0, 8))
    with pytest.raises(ValueError):
        CLAHEContrast(n_bins=1)


def test_to_dict_round_trip():
    t = CLAHEContrast(clip_limit=(1.0, 3.0), grid=(8, 8), n_bins=256, p=0.7)
    d = t.to_dict()
    assert d["name"] == "CLAHEContrast"
    t2 = CLAHEContrast(**d["params"])
    assert t2.clip_range == t.clip_range
    assert t2.grid == t.grid
    assert t2.n_bins == t.n_bins
