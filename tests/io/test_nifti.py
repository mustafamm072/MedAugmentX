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


@pytest.mark.parametrize("unit,factor", [("mm", 1), ("meter", 1000), ("micron", 0.001)])
def test_units_and_grid_changes_round_trip(tmp_path, unit, factor):
    from medaugmentx.transforms import CenterCrop, Pad, Resize

    affine = np.array([[1, 0.2, 0, 10], [0, 2, 0, 20], [0, 0, 3, -7], [0, 0, 0, 1]], float)
    image = nibabel.Nifti1Image(np.zeros((10, 12, 14), np.float32), affine)
    image.header.set_xyzt_units(unit)
    source = tmp_path / "source.nii.gz"
    nibabel.save(image, source)
    volume = load_nifti(str(source))
    volume = volume.replace(keypoints=np.array([[5.0, 6.0, 7.0]]))
    world = volume.index_to_world(volume.keypoints)
    np.testing.assert_allclose(volume.affine[:3, 3], affine[:3, 3] * factor)
    for transform in (CenterCrop((8, 9, 10)), Pad((12, 13, 14)), Resize((6, 7, 8))):
        volume = transform(volume)
        np.testing.assert_allclose(volume.index_to_world(volume.keypoints), world, atol=1e-8)
    destination = tmp_path / "out.nii.gz"
    save_nifti(volume, str(destination))
    restored = load_nifti(str(destination))
    np.testing.assert_allclose(restored.affine, volume.affine, rtol=1e-6, atol=1e-5)
    header = nibabel.load(destination).header
    assert header.get_xyzt_units()[0] == "mm"
    assert header["sform_code"] > 0
    assert header["qform_code"] == 0  # sheared geometry cannot be represented by qform


def test_label_io_and_registration_check(tmp_path):
    volume = MedVolume(
        np.zeros((4, 5, 6), np.float32), mask=np.full((4, 5, 6), 300, np.uint16), affine=np.eye(4)
    )
    image, mask = tmp_path / "image.nii", tmp_path / "mask.nii"
    save_nifti(volume, str(image), mask_path=str(mask))
    restored = load_nifti(str(image), mask_path=str(mask))
    np.testing.assert_array_equal(restored.mask, volume.mask)
    shifted = nibabel.Nifti1Image(np.zeros((6, 5, 4), np.uint8), np.diag([1, 1, 2, 1]))
    shifted.header.set_xyzt_units("mm")
    nibabel.save(shifted, mask)
    with pytest.raises(ValueError, match="patient-space grid"):
        load_nifti(str(image), mask_path=str(mask))


def test_unknown_geometry_export_can_be_strict(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        save_nifti(
            MedVolume(np.zeros((4, 4))), str(tmp_path / "bad.nii"), allow_assumed_geometry=False
        )


@pytest.mark.parametrize("dtype", [None, np.float32])
def test_loaded_arrays_own_data_and_release_source_mappings(tmp_path, dtype):
    volume = MedVolume(
        np.arange(120, dtype=np.float32).reshape(4, 5, 6),
        mask=np.full((4, 5, 6), 300, np.uint16),
        affine=np.eye(4),
    )
    image_path, mask_path = tmp_path / "image.nii", tmp_path / "mask.nii"
    save_nifti(volume, str(image_path), mask_path=str(mask_path))

    restored = load_nifti(str(image_path), dtype=dtype, mask_path=str(mask_path))

    # NIfTI disk order transposes into a C-contiguous memmap. Merely making
    # it contiguous retains the mapping (and a Windows file lock).
    assert restored.image.flags.owndata
    assert restored.mask.flags.owndata
    replacement = volume.replace(image=np.zeros_like(volume.image), mask=np.zeros_like(volume.mask))
    save_nifti(replacement, str(image_path), mask_path=str(mask_path))
    np.testing.assert_array_equal(restored.image, volume.image)
    np.testing.assert_array_equal(restored.mask, volume.mask)
