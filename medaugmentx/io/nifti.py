"""NIfTI I/O with explicit array-to-RAS millimetre geometry."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from medaugmentx.core.physical import assumed_affine, axis_permutation
from medaugmentx.core.volume import MedVolume


def _import_nibabel():
    try:
        import nibabel
    except ImportError as exc:
        raise ImportError(
            "nibabel is required for NIfTI I/O. Install with: pip install 'medaugmentx[nifti]'"
        ) from exc
    return nibabel


def load_nifti(
    path: str, dtype: Any | None = np.float32, *, mask_path: str | None = None
) -> MedVolume:
    """Load 2D/3D NIfTI data and optionally a registered integer label map.

    File axes are reversed into array order. Spatial units are converted to mm;
    unknown units are explicitly assumed mm with a warning. Source orientation
    is preserved, not canonicalized. Four-dimensional time/channel data is
    unsupported except for a trailing singleton dimension.

    Args:
        path: NIfTI image filename.
        dtype: Output intensity dtype, or None to retain the proxy's scaled dtype.
        mask_path: Optional label-map NIfTI on exactly the same physical grid.
    """
    nib = _import_nibabel()
    source = nib.load(str(path))
    data = np.asanyarray(source.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim not in (2, 3):
        raise ValueError(f"NIfTI file has unsupported ndim={data.ndim} (shape={data.shape})")
    if dtype is not None:
        data = data.astype(dtype, copy=False)
    unit = source.header.get_xyzt_units()[0]
    if unit not in ("mm", "meter", "micron", "unknown"):
        raise ValueError(f"Unsupported NIfTI spatial units: {unit}")
    if unit == "unknown":
        warnings.warn(
            "NIfTI spatial units are unknown; assuming millimetres", UserWarning, stacklevel=2
        )
    factor = {"mm": 1.0, "meter": 1000.0, "micron": 0.001, "unknown": 1.0}[unit]
    native = np.array(source.affine, dtype=float)
    native[:3, :] *= factor
    affine = native @ axis_permutation(data.ndim)
    metadata = {
        "source": str(path),
        "modality": None,
        "affine": native,
        "nifti_spatial_units": unit,
        "nifti_qform_code": int(source.header["qform_code"]),
        "nifti_sform_code": int(source.header["sform_code"]),
    }
    volume = MedVolume(np.ascontiguousarray(data.transpose()), metadata=metadata, affine=affine)
    if mask_path is not None:
        mask_volume = load_nifti(mask_path, dtype=None)
        assert mask_volume.affine is not None and volume.affine is not None
        if mask_volume.shape != volume.shape or not np.allclose(
            mask_volume.affine, volume.affine, rtol=1e-5, atol=1e-5
        ):
            raise ValueError("NIfTI image and mask must have the same shape and patient-space grid")
        volume = volume.replace(mask=mask_volume.image)
    return volume


def save_nifti(
    volume: MedVolume,
    path: str,
    *,
    mask_path: str | None = None,
    allow_assumed_geometry: bool = True,
) -> None:
    """Export current geometry, setting mm units and coherent qform/sform.

    Shear is preserved in sform; qform is disabled when it cannot represent the
    grid without approximation. Unknown patient geometry uses an explicitly
    assumed axis-aligned grid with a warning, or raises in strict export mode.

    Args:
        volume: Volume to write.
        path: Image output filename.
        mask_path: Optional separate integer-label output filename.
        allow_assumed_geometry: Permit axis-aligned export when affine is unknown.
    """
    nib = _import_nibabel()
    if mask_path is not None and volume.mask is None:
        raise ValueError("mask_path requires a mask")
    if mask_path is not None and str(mask_path) == str(path):
        raise ValueError("image and mask output paths must differ")
    affine = volume.affine
    if affine is None:
        if not allow_assumed_geometry:
            raise ValueError("Patient geometry is unknown; supply an affine before export")
        warnings.warn(
            "Exporting unknown patient geometry on an assumed axis-aligned grid",
            UserWarning,
            stacklevel=2,
        )
        affine = assumed_affine(volume.spacing)
    native = affine @ axis_permutation(volume.ndim)
    gram = native[:3, :3].T @ native[:3, :3]
    orthogonal = np.allclose(gram, np.diag(np.diag(gram)), rtol=1e-6, atol=1e-8)

    def write(array: np.ndarray, destination: str) -> None:
        img = nib.Nifti1Image(array.transpose(), native, dtype=array.dtype)
        img.header.set_xyzt_units("mm")
        img.set_sform(native, code=volume.metadata.get("nifti_sform_code", 1) or 1)
        if orthogonal:
            img.set_qform(native, code=volume.metadata.get("nifti_qform_code", 1) or 1)
        else:
            img.set_qform(None, code=0)
        nib.save(img, str(destination))

    write(volume.image, path)
    if mask_path is not None:
        assert volume.mask is not None
        labels = volume.mask
        if labels.dtype.kind not in "iu":
            labels = labels.astype(np.int64)
        write(labels, mask_path)
