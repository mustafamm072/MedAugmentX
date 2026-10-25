import uuid
import warnings

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from medaugmentx.io import load_dicom_series

pytestmark = pytest.mark.io


def _write_dicom_slice(path, pixels: np.ndarray, *, series_uid, position: float, modality="CT"):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    # Match the dataset encoding to TransferSyntaxUID on pydicom 2.x as well
    # as 3.x; older writers do not infer it from file_meta on initial save.
    ds = FileDataset(
        str(path),
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
        is_implicit_VR=False,
        is_little_endian=True,
    )
    ds.PatientID = "test"
    ds.Modality = modality
    ds.Manufacturer = "TestVendor"
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = "2.25." + str(uuid.uuid5(uuid.NAMESPACE_OID, series_uid).int)
    ds.FrameOfReferenceUID = "2.25." + str(uuid.uuid5(uuid.NAMESPACE_DNS, series_uid).int)
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    ds.Rows = pixels.shape[0]
    ds.Columns = pixels.shape[1]
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 1.0
    ds.ImagePositionPatient = [0.0, 0.0, float(position)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0

    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    ds.save_as(str(path))


def test_load_3d_series(tmp_path):
    series_uid = generate_uid()
    for i in range(5):
        slice_pixels = np.full((16, 16), 1024 + i * 100, dtype=np.uint16)
        _write_dicom_slice(
            tmp_path / f"slice_{i}.dcm", slice_pixels, series_uid=series_uid, position=i * 1.5
        )

    vol = load_dicom_series(str(tmp_path))
    assert vol.image.shape == (5, 16, 16)
    # spacing: z = 1.5 (median of position diffs), y/x = 0.5
    assert abs(vol.spacing[0] - 1.5) < 1e-6
    assert vol.spacing[1] == 0.5 and vol.spacing[2] == 0.5
    # rescale applied: pixel 1024 -> 0 HU, pixel 1124 -> 100 HU
    assert vol.image[0].mean() == pytest.approx(0.0, abs=1e-3)
    assert vol.image[-1].mean() == pytest.approx(400.0, abs=1e-3)
    assert vol.metadata["modality"] == "CT"
    assert vol.metadata["vendor"] == "TestVendor"


def test_multiple_series_rejected(tmp_path):
    s1 = generate_uid()
    s2 = generate_uid()
    _write_dicom_slice(
        tmp_path / "a.dcm", np.zeros((8, 8), dtype=np.uint16), series_uid=s1, position=0
    )
    _write_dicom_slice(
        tmp_path / "b.dcm", np.zeros((8, 8), dtype=np.uint16), series_uid=s2, position=1
    )
    with pytest.raises(ValueError, match="multiple DICOM series"):
        load_dicom_series(str(tmp_path))


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dicom_series(str(tmp_path / "does-not-exist"))


def test_instance_numbers_are_not_physical_spacing(tmp_path):
    uid = generate_uid()
    for i in range(2):
        path = tmp_path / f"{i}.dcm"
        _write_dicom_slice(path, np.zeros((4, 4)), series_uid=uid, position=i)
        ds = pydicom.dcmread(path)
        del ds.ImagePositionPatient
        del ds.ImageOrientationPatient
        ds.InstanceNumber = i * 10
        ds.SpacingBetweenSlices = 2.5
        ds.save_as(path)
    with pytest.warns(UserWarning, match="unknown"):
        volume = load_dicom_series(str(tmp_path), strict=False)
    assert volume.spacing[0] == 2.5
    assert volume.affine is None


def test_missing_rescale_tags_do_not_inherit_other_slice(tmp_path):
    uid = generate_uid()
    for i in range(2):
        path = tmp_path / f"{i}.dcm"
        _write_dicom_slice(path, np.full((4, 4), 1024), series_uid=uid, position=i)
        if i == 1:
            ds = pydicom.dcmread(path)
            del ds.RescaleIntercept
            del ds.RescaleSlope
            ds.save_as(path)
    volume = load_dicom_series(str(tmp_path))
    assert volume.image[0, 0, 0] == 0
    assert volume.image[1, 0, 0] == 1024


@pytest.mark.parametrize("positions", [(0, 0, 1), (0, 1, 4)])
def test_nonregular_slice_positions_rejected(tmp_path, positions):
    uid = generate_uid()
    for i, position in enumerate(positions):
        _write_dicom_slice(
            tmp_path / f"{i}.dcm", np.zeros((4, 4)), series_uid=uid, position=position
        )
    with pytest.raises(ValueError, match="distinct|irregular"):
        load_dicom_series(str(tmp_path))


@pytest.mark.parametrize(
    "tag,value",
    [
        ("ImageOrientationPatient", [0, 1, 0, 1, 0, 0]),
        ("PixelSpacing", [0.25, 0.5]),
        ("SamplesPerPixel", 3),
    ],
)
def test_inconsistent_or_color_dicom_rejected(tmp_path, tag, value):
    uid = generate_uid()
    for i in range(2):
        path = tmp_path / f"{i}.dcm"
        _write_dicom_slice(path, np.zeros((4, 4)), series_uid=uid, position=i)
        if i == 1:
            ds = pydicom.dcmread(path)
            setattr(ds, tag, value)
            ds.save_as(path)
    with pytest.raises(ValueError, match="inconsistent|monochrome"):
        load_dicom_series(str(tmp_path))


def test_fixture_encoding_matches_transfer_syntax_on_rewrite(tmp_path):
    path = tmp_path / "slice.dcm"
    pixels = np.arange(16, dtype=np.uint16).reshape(4, 4)
    _write_dicom_slice(path, pixels, series_uid=generate_uid(), position=0)
    # pydicom warns (rather than raises) when the declared transfer syntax and
    # the on-disk encoding disagree, so promote that warning to a failure here.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ds = pydicom.dcmread(path)
        assert ds.SOPClassUID == ds.file_meta.MediaStorageSOPClassUID
        assert ds.is_implicit_VR is False
        assert ds.is_little_endian is True
        ds.InstanceNumber = 2
        ds.save_as(path)
        restored = pydicom.dcmread(path)
        np.testing.assert_array_equal(restored.pixel_array, pixels)


def test_regular_shear_preserves_all_patient_coordinates(tmp_path):
    uid = generate_uid()
    for i in range(3):
        path = tmp_path / f"{i}.dcm"
        _write_dicom_slice(path, np.full((4, 5), 1024), series_uid=uid, position=i)
        ds = pydicom.dcmread(path)
        ds.ImagePositionPatient = [10 + 0.5 * i, 20, 30 + i]
        ds.save_as(path)
    volume = load_dicom_series(str(tmp_path))
    np.testing.assert_allclose(
        volume.index_to_world([[0, 0, 0], [2, 1, 3]]), [[-10, -20, 30], [-12.5, -20.5, 32]]
    )
    assert volume.spacing[0] == pytest.approx(np.sqrt(1.25))
    assert "patient_id" not in volume.metadata
    assert (
        load_dicom_series(str(tmp_path), include_patient_id=True).metadata["patient_id"] == "test"
    )


@pytest.mark.parametrize(
    "tag,value,message",
    [
        ("NumberOfFrames", 2, "multiframe"),
        ("ImageOrientationPatient", [2, 0, 0, 0, 0.5, 0], "orthogonal unit"),
        ("SOPClassUID", "1.2.840.10008.5.1.4.1.1.4.1", "SOP class"),
        ("RescaleSlope", 0, "rescale"),
    ],
)
def test_unsupported_or_invalid_single_file_fails(tmp_path, tag, value, message):
    path = tmp_path / "input.dcm"
    _write_dicom_slice(path, np.zeros((4, 4)), series_uid=generate_uid(), position=0)
    ds = pydicom.dcmread(path)
    setattr(ds, tag, value)
    ds.save_as(path)
    with pytest.raises(ValueError, match=message):
        load_dicom_series(str(path))


def test_unreadable_candidates_are_reported(tmp_path):
    _write_dicom_slice(tmp_path / "ok.dcm", np.zeros((4, 4)), series_uid=generate_uid(), position=0)
    (tmp_path / "bad.dcm").write_text("corrupt")
    with pytest.raises(ValueError, match="Unreadable"):
        load_dicom_series(str(tmp_path))
    with pytest.warns(UserWarning, match="Skipped"):
        volume = load_dicom_series(str(tmp_path), strict=False)
    assert len(volume.metadata["skipped_files"]) == 1
