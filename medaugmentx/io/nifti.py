"""NIfTI reader and writer (single-file ``.nii`` / ``.nii.gz``)."""
from __future__ import annotations

from typing import Any

import numpy as np

from medaugmentx.core.volume import MedVolume


def _import_nibabel():
    try:
        import nibabel  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise ImportError(
            "nibabel is required for NIfTI I/O. Install with: pip install 'medaugmentx[nifti]'"
        ) from exc
    return nibabel


def load_nifti(path: str, dtype: Any | None = np.float32) -> MedVolume:
    """Load a NIfTI file (``.nii`` / ``.nii.gz``) into a ``MedVolume``.

    The image data is transposed from NIfTI's ``(X, Y, Z)`` ordering to the
    ``(Z, Y, X)`` ordering used throughout MedAugment, and the spacing
    tuple is reordered to match.

    Args:
        path: Path to the NIfTI file.
        dtype: Output dtype; pass ``None`` to keep the on-disk dtype.

    Returns:
        A :class:`MedVolume` with ``spacing=(z_mm, y_mm, x_mm)`` and a
        ``metadata`` dict including the original 4×4 affine matrix.
    """
    nib = _import_nibabel()
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if dtype is not None:
        data = data.astype(dtype, copy=False)

    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim not in (2, 3):
        raise ValueError(f"NIfTI file has unsupported ndim={data.ndim} (shape={data.shape})")

    zooms = img.header.get_zooms()[: data.ndim]
    if data.ndim == 3:
        # NIfTI stores (X, Y, Z); MedVolume uses (Z, Y, X).
        data = np.transpose(data, (2, 1, 0))
        spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    else:
        # 2D: (X, Y) -> (Y, X)
        data = np.transpose(data, (1, 0))
        spacing = (float(zooms[1]), float(zooms[0]))

    metadata = {
        "source": str(path),
        "modality": None,
        "affine": np.array(img.affine, dtype=np.float64),
        "nifti_zooms": tuple(float(z) for z in img.header.get_zooms()),
    }
    return MedVolume(image=np.ascontiguousarray(data), spacing=spacing, metadata=metadata)


def save_nifti(volume: MedVolume, path: str) -> None:
    """Write a ``MedVolume`` to a NIfTI file.

    The image is transposed back to ``(X, Y, Z)`` ordering. If the volume's
    metadata contains an ``affine`` key (e.g. from :func:`load_nifti`) it
    is reused; otherwise an identity affine scaled by the spacing is
    written.

    Args:
        volume: Volume to write.
        path: Destination ``.nii`` or ``.nii.gz`` path.
    """
    nib = _import_nibabel()
    if volume.is_3d:
        data = np.transpose(volume.image, (2, 1, 0))
    else:
        data = np.transpose(volume.image, (1, 0))

    affine = volume.metadata.get("affine")
    if affine is None:
        affine = np.eye(4, dtype=np.float64)
        if volume.is_3d:
            affine[0, 0] = volume.spacing[2]
            affine[1, 1] = volume.spacing[1]
            affine[2, 2] = volume.spacing[0]
        else:
            affine[0, 0] = volume.spacing[1]
            affine[1, 1] = volume.spacing[0]

    img = nib.Nifti1Image(np.asarray(data), affine)
    if volume.is_3d:
        zooms = (volume.spacing[2], volume.spacing[1], volume.spacing[0])
    else:
        zooms = (volume.spacing[1], volume.spacing[0])
    img.header.set_zooms(zooms)
    nib.save(img, str(path))
