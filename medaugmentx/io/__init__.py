"""Unified I/O for medical image formats.

Each loader returns a :class:`~medaugmentx.core.volume.MedVolume` with
``spacing`` populated in millimetres and ``metadata`` carrying the
modality and any vendor-specific information that callers may need.

Optional dependencies:

- DICOM I/O requires ``pydicom`` (``pip install medaugmentx[dicom]``).
- NIfTI I/O requires ``nibabel`` (``pip install medaugmentx[nifti]``).

If a backend is missing the loader raises a clear :class:`ImportError`
when called, not at import time.
"""
from medaugmentx.io.dicom import load_dicom_series
from medaugmentx.io.nifti import load_nifti, save_nifti

__all__ = ["load_dicom_series", "load_nifti", "save_nifti"]
