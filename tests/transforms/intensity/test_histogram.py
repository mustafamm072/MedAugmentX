import numpy as np
import pytest

from medaugmentx.core import MedVolume
from medaugmentx.serialization import from_json, to_json
from medaugmentx.transforms import HistogramMatch


@pytest.fixture
def reference():
    return (np.random.default_rng(1).random((30, 30)).astype(np.float32) * 5.0 + 10.0)


@pytest.fixture
def vol2d():
    img = np.random.default_rng(2).random((40, 40)).astype(np.float32)
    mask = (img > 0.5).astype(np.uint8)
    return MedVolume(image=img, mask=mask)


def test_shape_dtype_preserved(vol2d, reference):
    out = HistogramMatch(reference=reference, seed=0)(vol2d)
    assert out.image.shape == vol2d.image.shape
    assert out.image.dtype == np.float32


def test_matches_reference_distribution(vol2d, reference):
    out = HistogramMatch(reference=reference, blend=1.0, seed=0)(vol2d)
    assert out.image.mean() == pytest.approx(float(reference.mean()), abs=0.5)


def test_none_reference_is_identity(vol2d):
    out = HistogramMatch(reference=None, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_blend_zero_is_identity(vol2d, reference):
    out = HistogramMatch(reference=reference, blend=0.0, seed=0)(vol2d)
    np.testing.assert_allclose(out.image, vol2d.image, atol=1e-5)


def test_mask_untouched(vol2d, reference):
    out = HistogramMatch(reference=reference, seed=0)(vol2d)
    np.testing.assert_array_equal(out.mask, vol2d.mask)


def test_deterministic(vol2d, reference):
    a = HistogramMatch(reference=reference, blend=(0.5, 1.0), seed=4)(vol2d)
    b = HistogramMatch(reference=reference, blend=(0.5, 1.0), seed=4)(vol2d)
    np.testing.assert_array_equal(a.image, b.image)


def test_p_zero_no_op(vol2d, reference):
    out = HistogramMatch(reference=reference, p=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_invalid_blend_raises():
    with pytest.raises(ValueError):
        HistogramMatch(blend=1.5)
    with pytest.raises(ValueError):
        HistogramMatch(blend=(0.5, 0.2))


def test_json_round_trip_preserves_reference(vol2d, reference):
    t = HistogramMatch(reference=reference, blend=1.0, n_quantiles=64, seed=0)
    rt = from_json(to_json(t))
    out_a = t(vol2d)
    out_b = rt(vol2d)
    np.testing.assert_allclose(out_a.image, out_b.image, atol=1e-4)


def test_none_reference_round_trips(vol2d):
    t = HistogramMatch(reference=None)
    rt = from_json(to_json(t))
    assert rt.reference is None
