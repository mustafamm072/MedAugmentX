import numpy as np
import pytest

nibabel = pytest.importorskip("nibabel")

from medaugmentx.core import MedVolume
from medaugmentx.io import load_nifti, save_nifti

pytestmark = pytest.mark.io


def test_save_and_load_3d_round_trip(tmp_path):
    img = np.random.default_rng(0).random((6, 8, 10)).astype(np.float32)
    vol = MedVolume(image=img, spacing=(2.0, 0.5, 0.5))
    path = tmp_path / "vol.nii.gz"
    save_nifti(vol, str(path))
    reloaded = load_nifti(str(path))
    assert reloaded.image.shape == img.shape
    np.testing.assert_allclose(reloaded.image, img, atol=1e-5)
    # spacing within 0.001 mm
    for a, b in zip(reloaded.spacing, vol.spacing):
        assert abs(a - b) < 1e-3


def test_2d_round_trip(tmp_path):
    img = np.random.default_rng(0).random((8, 12)).astype(np.float32)
    vol = MedVolume(image=img, spacing=(0.7, 0.7))
    path = tmp_path / "img.nii.gz"
    save_nifti(vol, str(path))
    reloaded = load_nifti(str(path))
    assert reloaded.image.shape == img.shape
    np.testing.assert_allclose(reloaded.image, img, atol=1e-5)


def test_metadata_includes_affine(tmp_path):
    img = np.zeros((4, 4, 4), dtype=np.float32)
    vol = MedVolume(image=img, spacing=(1.0, 1.0, 1.0))
    path = tmp_path / "x.nii.gz"
    save_nifti(vol, str(path))
    reloaded = load_nifti(str(path))
    assert "affine" in reloaded.metadata
    assert reloaded.metadata["affine"].shape == (4, 4)
