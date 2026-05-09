import numpy as np
import pytest

from medaugment.core import MedVolume
from medaugment.transforms import BiasField


@pytest.fixture
def vol2d():
    img = np.ones((64, 64), dtype=np.float32)
    return MedVolume(image=img)


@pytest.fixture
def vol3d():
    img = np.ones((16, 32, 32), dtype=np.float32)
    return MedVolume(image=img)


def test_bias_field_changes_image_2d(vol2d):
    out = BiasField(alpha=0.3, seed=0)(vol2d)
    assert out.image.shape == vol2d.image.shape
    # Bias field must multiply by something ≠ 1 with nonzero alpha
    assert not np.allclose(out.image, vol2d.image)


def test_bias_field_changes_image_3d(vol3d):
    out = BiasField(alpha=0.3, seed=0)(vol3d)
    assert out.image.shape == vol3d.image.shape
    assert not np.allclose(out.image, vol3d.image)


def test_bias_field_zero_alpha_is_identity(vol2d):
    out = BiasField(alpha=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_bias_field_output_is_positive(vol2d):
    out = BiasField(alpha=0.5, seed=1)(vol2d)
    # All values must be positive (multiplicative field > 0 always)
    assert out.image.min() > 0.0


def test_bias_field_mask_untouched():
    img = np.ones((32, 32), dtype=np.float32)
    mask = (img > 0.5).astype(np.uint8)
    vol = MedVolume(image=img, mask=mask)
    out = BiasField(alpha=0.3, seed=0)(vol)
    np.testing.assert_array_equal(out.mask, mask)


def test_bias_field_deterministic(vol3d):
    a = BiasField(alpha=0.3, seed=7)(vol3d)
    b = BiasField(alpha=0.3, seed=7)(vol3d)
    np.testing.assert_array_equal(a.image, b.image)


def test_bias_field_p_zero_no_op(vol2d):
    out = BiasField(alpha=0.3, p=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_bias_field_negative_alpha_raises():
    with pytest.raises(ValueError):
        BiasField(alpha=-0.1)


def test_bias_field_to_dict_round_trip(vol2d):
    t = BiasField(alpha=0.3, coarse_shape=4, order=1, p=0.8)
    d = t.to_dict()
    assert d["name"] == "BiasField"
    t2 = BiasField(**d["params"])
    assert t2.alpha == t.alpha
    assert t2.order == t.order
    assert t2.p == t.p
