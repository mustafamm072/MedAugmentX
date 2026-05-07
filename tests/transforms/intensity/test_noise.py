import numpy as np
import pytest

from medaugment.core import MedVolume
from medaugment.transforms import GaussianNoise, RicianNoise


def test_gaussian_noise_changes_image_but_preserves_shape():
    img = np.zeros((32, 32), dtype=np.float32)
    vol = MedVolume(image=img)
    out = GaussianNoise(std=0.05, seed=0)(vol)
    assert out.image.shape == img.shape
    assert out.image.std() > 0


def test_gaussian_noise_zero_std_is_identity():
    img = np.ones((32, 32), dtype=np.float32) * 0.5
    vol = MedVolume(image=img)
    out = GaussianNoise(std=0.0, seed=0)(vol)
    np.testing.assert_array_equal(out.image, img)


def test_gaussian_noise_clip_bounds_are_respected():
    img = np.full((16, 16), 0.5, dtype=np.float32)
    vol = MedVolume(image=img)
    out = GaussianNoise(std=2.0, clip=(0.0, 1.0), seed=0)(vol)
    assert out.image.min() >= 0.0 and out.image.max() <= 1.0


def test_gaussian_noise_negative_std_rejected():
    with pytest.raises(ValueError):
        GaussianNoise(std=-0.1)


def test_rician_noise_is_non_negative():
    img = np.zeros((32, 32), dtype=np.float32)
    vol = MedVolume(image=img)
    out = RicianNoise(std=0.1, seed=0)(vol)
    assert out.image.min() >= 0.0


def test_rician_seedable():
    img = np.full((32, 32), 0.5, dtype=np.float32)
    vol = MedVolume(image=img)
    a = RicianNoise(std=0.1, seed=42)(vol)
    b = RicianNoise(std=0.1, seed=42)(vol)
    np.testing.assert_array_equal(a.image, b.image)


def test_rician_at_low_signal_floor_is_above_zero():
    """Classic Rician property: mean of |N(0, σ) + jN(0, σ)| > 0."""
    img = np.zeros((128, 128), dtype=np.float32)
    vol = MedVolume(image=img)
    out = RicianNoise(std=0.1, seed=0)(vol)
    assert out.image.mean() > 0.05  # ~ σ * sqrt(π/2)
