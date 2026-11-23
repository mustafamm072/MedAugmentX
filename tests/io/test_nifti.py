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
    vol = MedVolume(image=img, affine=np.diag([1.0, 2.0, 3.0, 1.0]))
    path = tmp_path / "x.nii.gz"
    save_nifti(vol, str(path))
    reloaded = load_nifti(str(path))
    assert "affine" in reloaded.metadata
    assert reloaded.metadata["affine"].shape == (4, 4)
    np.testing.assert_allclose(reloaded.affine, vol.affine)


def test_unknown_geometry_stays_unknown_through_round_trip(tmp_path):
    vol = MedVolume(np.zeros((3, 4, 5), np.float32), spacing=(2.0, 0.5, 0.75))
    path = tmp_path / "unknown.nii.gz"
    with pytest.warns(UserWarning, match="assumed"):
        save_nifti(vol, str(path))
    header = nibabel.load(str(path)).header
    assert int(header["qform_code"]) == int(header["sform_code"]) == 0
    reloaded = load_nifti(str(path))
    assert reloaded.affine is None and "affine" not in reloaded.metadata
    np.testing.assert_allclose(reloaded.spacing, vol.spacing)
    mask_path = tmp_path / "mask.nii.gz"
    with pytest.warns(UserWarning, match="assumed"):
        save_nifti(
            vol.replace(mask=np.ones((3, 4, 5), np.uint8)),
            str(tmp_path / "i.nii.gz"),
            mask_path=str(mask_path),
        )
    paired = load_nifti(str(path), mask_path=str(mask_path))
    assert paired.mask.sum() == 60
    coded_mask = tmp_path / "coded_mask.nii.gz"
    nibabel.save(nibabel.Nifti1Image(np.ones((6, 4, 3), np.uint8), np.eye(4)), str(coded_mask))
    with pytest.raises(ValueError, match="coded qform or sform"):
        load_nifti(str(path), mask_path=str(coded_mask), resample_mask=True)


def test_export_labels_both_forms_with_the_written_space(tmp_path):
    scanner, mni = np.diag([1.0, 1.0, 1.0, 1.0]), np.diag([2.0, 2.0, 2.0, 1.0])
    img = nibabel.Nifti1Image(np.zeros((4, 4, 4), np.float32), None)
    img.set_qform(scanner, code=1)
    img.set_sform(mni, code=4)
    source = tmp_path / "source.nii.gz"
    nibabel.save(img, str(source))
    out = tmp_path / "out.nii.gz"
    save_nifti(load_nifti(str(source)), str(out))
    header = nibabel.load(str(out)).header
    assert int(header["sform_code"]) == int(header["qform_code"]) == 4
    np.testing.assert_allclose(header.get_qform(), mni)


@pytest.mark.parametrize("dtype", [np.bool_, np.float16])
def test_export_converts_unsupported_image_dtypes(tmp_path, dtype):
    vol = MedVolume(np.ones((3, 4, 5), dtype=dtype), affine=np.eye(4))
    path = tmp_path / "typed.nii.gz"
    save_nifti(vol, str(path))
    np.testing.assert_array_equal(load_nifti(str(path)).image, 1.0)


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


def test_opt_in_mask_resampling_preserves_uint64_labels_on_image_grid(tmp_path):
    affine = np.array([[0.1, 0, -0.7, 80], [0, 1.2, 0.1, -20], [2, 0, 0, 7], [0, 0, 0, 1]])
    image = MedVolume(np.zeros((3, 4, 5), dtype=np.float32), affine=affine)
    image_path, mask_path = tmp_path / "image.nii", tmp_path / "mask.nii.gz"
    save_nifti(image, str(image_path))
    mask_affine = affine.copy()
    mask_affine[:3, 3] += affine[:3, :3].sum(axis=1)
    labels = np.full((2, 3, 4), 2**63 + 1, dtype=np.uint64)
    labels[0, 1, 2] = 2**64 - 1
    save_nifti(MedVolume(labels, affine=mask_affine), str(mask_path))

    with pytest.raises(ValueError, match="patient-space grid"):
        load_nifti(str(image_path), mask_path=str(mask_path))
    output = load_nifti(str(image_path), mask_path=str(mask_path), resample_mask=True)
    expected = np.zeros(image.shape, dtype=labels.dtype)
    expected[1:, 1:, 1:] = labels
    np.testing.assert_array_equal(output.mask, expected)
    assert output.mask.dtype == labels.dtype
    np.testing.assert_array_equal(output.affine, load_nifti(str(image_path)).affine)


def test_strict_grid_match_uses_absolute_patient_distance(tmp_path):
    image_path, mask_path = tmp_path / "image.nii", tmp_path / "mask.nii"
    affine = np.eye(4)
    affine[0, 3] = 2000
    save_nifti(MedVolume(np.zeros((3, 4)), affine=affine), str(image_path))
    affine[0, 3] += 0.015
    save_nifti(MedVolume(np.zeros((3, 4)), affine=affine), str(mask_path))
    with pytest.raises(ValueError, match="patient-space grid"):
        load_nifti(str(image_path), mask_path=str(mask_path))


@pytest.mark.parametrize(
    "labels, shift, match",
    [
        (np.ones((2, 2, 2), dtype=np.uint8), 10, "removed all nonzero"),
        (np.full((2, 2, 2), 0.5, dtype=np.float32), 10, "integer-valued"),
    ],
)
def test_mask_resampling_rejects_total_label_loss_and_invalid_labels(
    tmp_path, labels, shift, match
):
    image_path, mask_path = tmp_path / "image.nii", tmp_path / "mask.nii"
    save_nifti(MedVolume(np.zeros((4, 4, 4)), affine=np.eye(4)), str(image_path))
    affine = np.eye(4)
    affine[0, 3] = shift
    save_nifti(MedVolume(labels, affine=affine), str(mask_path))
    with pytest.raises(ValueError, match=match):
        load_nifti(str(image_path), mask_path=str(mask_path), resample_mask=True)


def test_opt_in_mask_alignment_requires_coded_geometry_and_matching_ndim(tmp_path):
    image_path, mask_path = tmp_path / "image.nii", tmp_path / "mask.nii"
    save_nifti(MedVolume(np.zeros((4, 4, 4)), affine=np.eye(4)), str(image_path))
    mask = nibabel.Nifti1Image(np.zeros((3, 3, 3), dtype=np.uint8), None)
    mask.header.set_xyzt_units("mm")
    nibabel.save(mask, mask_path)
    with pytest.raises(ValueError, match="coded qform or sform"):
        load_nifti(str(image_path), mask_path=str(mask_path), resample_mask=True)
    save_nifti(MedVolume(np.zeros((4, 4)), affine=np.eye(4)), str(mask_path))
    with pytest.raises(ValueError, match="dimensionality"):
        load_nifti(str(image_path), mask_path=str(mask_path), resample_mask=True)


def test_mask_resampling_requires_a_mask_path():
    with pytest.raises(ValueError, match="requires mask_path"):
        load_nifti("unused.nii", resample_mask=True)


def test_nifti_exports_wide_nonnegative_float_labels_as_uint64(tmp_path):
    image_path, mask_path = tmp_path / "image.nii", tmp_path / "mask.nii"
    integer_labels = np.array([[0, 2**63], [1, 2**63 + 2**16]], dtype=np.uint64)
    volume = MedVolume(np.zeros((2, 2)), integer_labels.astype(np.float64), affine=np.eye(4))
    save_nifti(volume, str(image_path), mask_path=str(mask_path))
    output = load_nifti(str(image_path), mask_path=str(mask_path))
    assert output.mask.dtype == np.uint64
    np.testing.assert_array_equal(output.mask, integer_labels)


def test_nifti_explicit_channel_selection_prevents_silent_channel_loss(tmp_path):
    image_path = tmp_path / "selected.nii"
    volume = MedVolume(np.zeros((2, 2)), affine=np.eye(4), channels={"flair": np.ones((2, 2))})
    with pytest.raises(ValueError, match="select_channel"):
        save_nifti(volume, str(image_path))
    assert not image_path.exists()
    save_nifti(volume.select_channel("flair"), str(image_path))
    np.testing.assert_array_equal(load_nifti(str(image_path)).image, np.ones((2, 2)))


@pytest.mark.parametrize("labels", [[0, 2.0**64], [-1, 2.0**63], [0, -(2.0**64)]])
def test_nifti_rejects_unrepresentable_float_labels_before_writing(tmp_path, labels):
    image_path, mask_path = tmp_path / "image.nii", tmp_path / "mask.nii"
    volume = MedVolume(np.zeros((2, 2)), np.tile(labels, (2, 1)), affine=np.eye(4))
    with pytest.raises(ValueError, match="lossless export"):
        save_nifti(volume, str(image_path), mask_path=str(mask_path))
    assert not image_path.exists()
    assert not mask_path.exists()
