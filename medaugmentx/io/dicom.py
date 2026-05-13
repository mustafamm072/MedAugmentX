"""DICOM series loader.

Phase 1 ships a single, vendor-agnostic loader that handles the common case:
a directory of single-frame DICOMs (one per slice) sharing a SeriesInstanceUID.
Vendor-specific multi-frame DBT parsers (Hologic, GE, Siemens) land in
Phase 2 — see ``docs/MILESTONES.md``.
"""
from __future__ import annotations

import os

import numpy as np

from medaugmentx.core.volume import MedVolume


def _import_pydicom():
    try:
        import pydicom  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise ImportError(
            "pydicom is required for DICOM I/O. Install with: pip install 'medaugmentx[dicom]'"
        ) from exc
    return pydicom


def _safe_get(ds, key, default=None):
    try:
        v = ds.get(key, default)
    except Exception:
        return default
    if v is None:
        return default
    return v


def _slice_position(ds) -> float | None:
    """Best-effort slice ordering key.

    Prefer ImagePositionPatient projected onto the slice normal; fall back
    to SliceLocation, then InstanceNumber.
    """
    ipp = _safe_get(ds, "ImagePositionPatient")
    iop = _safe_get(ds, "ImageOrientationPatient")
    if ipp is not None and iop is not None and len(iop) == 6:
        row = np.array(iop[:3], dtype=np.float64)
        col = np.array(iop[3:], dtype=np.float64)
        normal = np.cross(row, col)
        return float(np.dot(np.array(ipp, dtype=np.float64), normal))
    sl = _safe_get(ds, "SliceLocation")
    if sl is not None:
        return float(sl)
    inst = _safe_get(ds, "InstanceNumber")
    if inst is not None:
        return float(inst)
    return None


def _list_dicom_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    files: list[str] = []
    for root, _, names in os.walk(path):
        for name in names:
            if name.startswith("."):
                continue
            lower = name.lower()
            if lower.endswith(".dcm") or lower.endswith(".dicom"):
                files.append(os.path.join(root, name))
            elif "." not in name:
                # Many PACS exports ship extension-less files. Cheap to accept.
                files.append(os.path.join(root, name))
    if not files:
        raise FileNotFoundError(f"No DICOM-like files found under {path}")
    return files


def load_dicom_series(path: str) -> MedVolume:
    """Load a directory of single-frame DICOM files into a 3D ``MedVolume``.

    Slices are sorted by image position projected onto the slice normal;
    voxel spacing is read from ``PixelSpacing`` (in-plane) and the
    inter-slice distance is computed from sorted positions when possible,
    falling back to ``SliceThickness``.

    Pixel intensities are rescaled with the standard
    ``RescaleSlope * pixel + RescaleIntercept`` (HU for CT, scanner units
    elsewhere). A best-effort modality string is stored in ``metadata``.

    Args:
        path: Directory containing the DICOM files (or a single file).

    Returns:
        A :class:`MedVolume` with ``spacing=(z_mm, y_mm, x_mm)`` and
        ``metadata`` populated with ``modality``, ``vendor``,
        ``patient_id``, and ``series_uid`` when available.

    Raises:
        ImportError: If ``pydicom`` is not installed.
        FileNotFoundError: If the path does not exist or contains no DICOMs.
        ValueError: If files belong to multiple SeriesInstanceUIDs.
    """
    pydicom = _import_pydicom()
    files = _list_dicom_files(path)

    datasets = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=False)
        except Exception:
            continue
        if not hasattr(ds, "PixelData"):
            continue
        datasets.append((f, ds))

    if not datasets:
        raise FileNotFoundError(f"No readable DICOM pixel data under {path}")

    series_uids = {str(_safe_get(ds, "SeriesInstanceUID", "")) for _, ds in datasets}
    series_uids.discard("")
    if len(series_uids) > 1:
        raise ValueError(
            f"Path {path} contains multiple DICOM series ({len(series_uids)} UIDs); "
            "split them into separate folders before loading."
        )

    if len(datasets) == 1:
        # Single 2D image (e.g. mammography projection) or multi-frame.
        _, ds = datasets[0]
        pixels = ds.pixel_array
        slope = float(_safe_get(ds, "RescaleSlope", 1.0))
        intercept = float(_safe_get(ds, "RescaleIntercept", 0.0))
        image = pixels.astype(np.float32) * slope + intercept

        if image.ndim == 2:
            ps = _safe_get(ds, "PixelSpacing", [1.0, 1.0])
            spacing = (float(ps[0]), float(ps[1]))
        else:
            ps = _safe_get(ds, "PixelSpacing", [1.0, 1.0])
            thickness = float(_safe_get(ds, "SliceThickness", 1.0))
            spacing = (thickness, float(ps[0]), float(ps[1]))

        metadata = _build_metadata(ds, source=path)
        return MedVolume(image=image, spacing=spacing, metadata=metadata)

    # Multi-file series: sort by slice position.
    sortable = []
    for f, ds in datasets:
        pos = _slice_position(ds)
        sortable.append((pos if pos is not None else 0.0, f, ds))
    sortable.sort(key=lambda t: t[0])

    ref_ds = sortable[0][2]
    slope = float(_safe_get(ref_ds, "RescaleSlope", 1.0))
    intercept = float(_safe_get(ref_ds, "RescaleIntercept", 0.0))

    slices: list[np.ndarray] = []
    positions: list[float] = []
    for pos, _, ds in sortable:
        s = ds.pixel_array.astype(np.float32) * float(_safe_get(ds, "RescaleSlope", slope))
        s = s + float(_safe_get(ds, "RescaleIntercept", intercept))
        if s.ndim != 2:
            raise ValueError(f"Expected 2D slices, got shape {s.shape}")
        slices.append(s)
        positions.append(pos)

    image = np.stack(slices, axis=0).astype(np.float32, copy=False)

    ps = _safe_get(ref_ds, "PixelSpacing", [1.0, 1.0])
    in_plane = (float(ps[0]), float(ps[1]))

    if len(positions) >= 2:
        diffs = np.diff(positions)
        z_spacing = float(np.median(np.abs(diffs))) or float(_safe_get(ref_ds, "SliceThickness", 1.0))
    else:
        z_spacing = float(_safe_get(ref_ds, "SliceThickness", 1.0))

    metadata = _build_metadata(ref_ds, source=path)
    metadata["num_slices"] = len(slices)
    return MedVolume(image=image, spacing=(z_spacing, in_plane[0], in_plane[1]), metadata=metadata)


def _build_metadata(ds, *, source: str) -> dict:
    return {
        "source": str(source),
        "modality": str(_safe_get(ds, "Modality", "")) or None,
        "vendor": str(_safe_get(ds, "Manufacturer", "")) or None,
        "patient_id": str(_safe_get(ds, "PatientID", "")) or None,
        "study_uid": str(_safe_get(ds, "StudyInstanceUID", "")) or None,
        "series_uid": str(_safe_get(ds, "SeriesInstanceUID", "")) or None,
        "rescale_slope": float(_safe_get(ds, "RescaleSlope", 1.0)),
        "rescale_intercept": float(_safe_get(ds, "RescaleIntercept", 0.0)),
    }
