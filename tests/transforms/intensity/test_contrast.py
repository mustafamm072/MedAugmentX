import numpy as np
import pytest

from medaugmentx.core import MedVolume
from medaugmentx.transforms import GammaCorrection


def test_gamma_one_is_near_identity():
    img = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(16, 16)
    vol = MedVolume(image=img)
    out = GammaCorrection(gamma=1.0)(vol)
    np.testing.assert_allclose(out.image, img, atol=1e-5)


def test_gamma_preserves_min_and_max():
    img = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(16, 16)
    vol = MedVolume(image=img)
    out = GammaCorrection(gamma=(0.5, 2.0), seed=0)(vol)
    assert abs(float(out.image.min()) - 0.0) < 1e-5
    assert abs(float(out.image.max()) - 1.0) < 1e-5


def test_constant_image_unchanged():
    img = np.full((8, 8), 0.42, dtype=np.float32)
    vol = MedVolume(image=img)
    out = GammaCorrection(gamma=2.0)(vol)
    np.testing.assert_array_equal(out.image, img)


def test_invert_flag_is_symmetric():
    img = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    vol = MedVolume(image=img)
    a = GammaCorrection(gamma=1.0, invert=True)(vol)
    np.testing.assert_allclose(a.image, img, atol=1e-5)


def test_negative_gamma_rejected():
    with pytest.raises(ValueError):
        GammaCorrection(gamma=(-1.0, 1.0))


def test_seedable_reproducibility():
    img = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    vol = MedVolume(image=img)
    a = GammaCorrection(gamma=(0.5, 1.5), seed=42)(vol)
    b = GammaCorrection(gamma=(0.5, 1.5), seed=42)(vol)
    np.testing.assert_array_equal(a.image, b.image)
