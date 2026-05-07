import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from medaugment.io import load_dicom_series

pytestmark = pytest.mark.io


def _write_dicom_slice(path, pixels: np.ndarray, *, series_uid, position: float, modality="CT"):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientID = "test"
    ds.Modality = modality
    ds.Manufacturer = "TestVendor"
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = generate_uid()
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
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    ds.save_as(str(path))


def test_load_3d_series(tmp_path):
    series_uid = generate_uid()
    for i in range(5):
        slice_pixels = np.full((16, 16), 1024 + i * 100, dtype=np.uint16)
        _write_dicom_slice(tmp_path / f"slice_{i}.dcm", slice_pixels, series_uid=series_uid, position=i * 1.5)

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
    _write_dicom_slice(tmp_path / "a.dcm", np.zeros((8, 8), dtype=np.uint16), series_uid=s1, position=0)
    _write_dicom_slice(tmp_path / "b.dcm", np.zeros((8, 8), dtype=np.uint16), series_uid=s2, position=1)
    with pytest.raises(ValueError, match="multiple DICOM series"):
        load_dicom_series(str(tmp_path))


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dicom_series(str(tmp_path / "does-not-exist"))
