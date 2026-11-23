"""NIfTI I/O with explicit array-to-RAS millimetre geometry."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from medaugmentx.core.physical import assumed_affine, axis_permutation
from medaugmentx.core.resampling import grid_pull, grids_match, sample_grid_mask
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
    path: str,
    dtype: Any | None = np.float32,
    *,
    mask_path: str | None = None,
    resample_mask: bool = False,
) -> MedVolume:
    """Load 2D/3D NIfTI data and optionally a registered integer label map.

    File axes are reversed into array order. Spatial units are converted to mm;
    unknown units are explicitly assumed mm with a warning. Source orientation
    is preserved, not canonicalized. Files with neither a coded qform nor a
    coded sform carry no patient coordinates, so they load with ``affine=None``
    and spacing from the voxel sizes. Four-dimensional time/channel data is
    unsupported except for a trailing singleton dimension.
    Returned image/mask arrays own their data and retain no source-file mappings.

    Args:
        path: NIfTI image filename.
        dtype: Output intensity dtype, or None to retain the proxy's scaled dtype.
        mask_path: Optional registered label-map NIfTI. The default requires
            matching shape and world coordinates across the grid (within 1e-4 mm),
            or matching shape and spacing when either file has no coded geometry.
        resample_mask: For a grid mismatch, use the label map's header to sample
            its labels onto the image grid. Enable this only for files already
            sharing a patient coordinate system. Uses nearest-neighbour with
            background 0 outside the source mask. Requires a coded qform or
            sform in both files and raises if all nonzero labels disappear.
    """
    if resample_mask and mask_path is None:
        raise ValueError("resample_mask=True requires mask_path")
    nib = _import_nibabel()
    source = nib.load(str(path))
    data = np.asanyarray(source.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim not in (2, 3):
        raise ValueError(f"NIfTI file has unsupported ndim={data.ndim} (shape={data.shape})")
    unit = source.header.get_xyzt_units()[0]
    if unit not in ("mm", "meter", "micron", "unknown"):
        raise ValueError(f"Unsupported NIfTI spatial units: {unit}")
    if unit == "unknown":
        warnings.warn(
            "NIfTI spatial units are unknown; assuming millimetres", UserWarning, stacklevel=2
        )
    factor = {"mm": 1.0, "meter": 1000.0, "micron": 0.001, "unknown": 1.0}[unit]
    qform_code = int(source.header["qform_code"])
    sform_code = int(source.header["sform_code"])
    metadata: dict[str, Any] = {
        "source": str(path),
        "modality": None,
        "nifti_spatial_units": unit,
        "nifti_qform_code": qform_code,
        "nifti_sform_code": sform_code,
    }
    affine = None
    spacing: tuple[float, ...] = ()
    if qform_code or sform_code:
        native = np.array(source.affine, dtype=float)
        native[:3, :] *= factor
        affine = native @ axis_permutation(data.ndim)
        metadata["affine"] = native
    else:
        # nibabel's fallback affine for uncoded files is invented, not patient
        # geometry; keep only the voxel sizes, in array order.
        zooms = source.header.get_zooms()[: data.ndim]
        spacing = tuple(float(z) * factor for z in reversed(zooms))
    # Transposing NIfTI's disk layout can already be C-contiguous, so
    # ascontiguousarray alone retains a memmap and locks the file on Windows.
    # Copy once, combining dtype conversion with ownership of the array.
    image = np.array(data.transpose(), dtype=dtype, order="C", copy=True)
    volume = MedVolume(image, spacing=spacing, metadata=metadata, affine=affine)
    if mask_path is not None:
        mask_volume = load_nifti(mask_path, dtype=None)
        labels = mask_volume.image
        if mask_volume.shape != volume.shape:
            same_grid = False
        elif mask_volume.affine is None or volume.affine is None:
            same_grid = np.allclose(mask_volume.spacing, volume.spacing, rtol=1e-5, atol=1e-6)
        else:
            same_grid = grids_match(mask_volume.affine, volume.affine, volume.shape)
        if not same_grid:
            if not resample_mask:
                raise ValueError(
                    "NIfTI image and mask must have the same shape and patient-space grid"
                )
            if mask_volume.ndim != volume.ndim:
                raise ValueError("NIfTI image and mask must have the same dimensionality")
            if any(
                not (v.metadata["nifti_qform_code"] or v.metadata["nifti_sform_code"])
                for v in (volume, mask_volume)
            ):
                raise ValueError(
                    "Resampling a NIfTI mask requires a coded qform or sform in both files"
                )
            # Validate before sampling so an invalid label cannot disappear
            # during a crop or downsampling.
            mask_volume.replace(mask=labels)
            assert mask_volume.affine is not None and volume.affine is not None
            _, matrix, offset = grid_pull(mask_volume.affine, volume.affine, volume.shape)
            aligned = sample_grid_mask(labels, matrix, offset, volume.shape)
            if not aligned.any() and labels.any():
                raise ValueError("Resampling removed all nonzero NIfTI mask labels")
            labels = aligned
        volume = volume.replace(mask=labels)
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
    grid without approximation. Both forms carry the code of the space the
    affine came from (the loaded sform's, else the qform's, else scanner).
    Unknown patient geometry uses an explicitly assumed axis-aligned grid with
    a warning and codes 0 (no patient coordinates), or raises in strict mode.
    Boolean and float16 images are written as uint8 and float32.

    Args:
        volume: Volume to write.
        path: Image output filename.
        mask_path: Optional separate integer-label output filename. Floating
            label maps must fit int64 or uint64 without changing any IDs.
        allow_assumed_geometry: Permit axis-aligned export when affine is unknown.
    """
    nib = _import_nibabel()
    if volume.channels:
        raise ValueError("save_nifti writes one intensity image; use volume.select_channel()")
    if mask_path is not None and volume.mask is None:
        raise ValueError("mask_path requires a mask")
    if mask_path is not None and str(mask_path) == str(path):
        raise ValueError("image and mask output paths must differ")
    labels = volume.mask if mask_path is not None else None
    if labels is not None and labels.dtype.kind not in "iu":
        low, high = int(labels.min()), int(labels.max())
        if -(2**63) <= low and high < 2**63:
            labels = labels.astype(np.int64)
        elif 0 <= low and high < 2**64:
            labels = labels.astype(np.uint64)
        else:
            raise ValueError("NIfTI mask labels must fit int64 or uint64 for lossless export")
    affine = volume.affine
    code = int(
        volume.metadata.get("nifti_sform_code") or volume.metadata.get("nifti_qform_code") or 1
    )
    if affine is None:
        code = 0
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
        img.set_sform(native, code=code)
        # One matrix is written to both forms, so both must name its space.
        if orthogonal and code:
            img.set_qform(native, code=code)
        else:
            img.set_qform(None, code=0)
        nib.save(img, str(destination))

    image = volume.image
    if image.dtype == np.bool_:
        image = image.astype(np.uint8)
    elif image.dtype.kind == "f" and image.dtype.itemsize < 4:
        image = image.astype(np.float32)
    write(image, path)
    if mask_path is not None:
        assert labels is not None
        write(labels, mask_path)
