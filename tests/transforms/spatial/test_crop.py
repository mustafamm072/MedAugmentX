import numpy as np
import pytest

from medaugmentx.core import MedVolume
from medaugmentx.transforms import AnatomicCrop


def test_crop_returns_correct_shape():
    img = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    vol = MedVolume(image=img)
    out = AnatomicCrop(size=(16, 16), seed=0)(vol)
    assert out.image.shape == (16, 16)


def test_crop_3d_returns_correct_shape():
    img = np.random.default_rng(0).random((10, 32, 32)).astype(np.float32)
    vol = MedVolume(image=img)
    out = AnatomicCrop(size=(5, 16, 16), seed=0)(vol)
    assert out.image.shape == (5, 16, 16)


def test_crop_size_too_large_clips_to_volume():
    img = np.zeros((8, 8), dtype=np.float32)
    vol = MedVolume(image=img)
    out = AnatomicCrop(size=(32, 32), seed=0)(vol)
    assert out.image.shape == (8, 8)


def test_crop_biases_to_foreground():
    img = np.zeros((32, 32), dtype=np.float32)
    img[28:30, 28:30] = 1.0  # foreground in the corner
    mask = (img > 0).astype(np.uint8)
    vol = MedVolume(image=img, mask=mask)
    # foreground_prob=1.0 -> always biased; the patch must contain a foreground voxel
    out = AnatomicCrop(size=(8, 8), foreground_prob=1.0, seed=0)(vol)
    assert (out.mask > 0).any()


def test_crop_seedable():
    img = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    vol = MedVolume(image=img)
    a = AnatomicCrop(size=(16, 16), seed=42)(vol)
    b = AnatomicCrop(size=(16, 16), seed=42)(vol)
    np.testing.assert_array_equal(a.image, b.image)


def test_size_ndim_must_match():
    vol = MedVolume(image=np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="ndim"):
        AnatomicCrop(size=(4, 4, 4))(vol)
