"""Strict single-frame DICOM loading with full LPS-to-RAS grid geometry."""

from __future__ import annotations

import os
import warnings

import numpy as np

from medaugmentx.core.volume import MedVolume

# Explicitly supported classic monochrome image storage classes. Enhanced,
# segmentation, secondary capture, and vendor multiframe objects are rejected.
SUPPORTED_SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.2": "CT",
    "1.2.840.10008.5.1.4.1.1.4": "MR",
    "1.2.840.10008.5.1.4.1.1.1": "CR",
    "1.2.840.10008.5.1.4.1.1.1.1": "DX",
    "1.2.840.10008.5.1.4.1.1.1.1.1": "DX",
    "1.2.840.10008.5.1.4.1.1.1.2": "MG",
    "1.2.840.10008.5.1.4.1.1.1.2.1": "MG",
}


def _import_pydicom():
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom is required; install 'medaugmentx[dicom]'") from exc
    return pydicom


def _safe_get(ds, key, default=None):
    value = ds.get(key, default)
    return default if value is None else value


def _slice_position(ds) -> float | None:
    if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "ImageOrientationPatient"):
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        return float(np.dot(ds.ImagePositionPatient, np.cross(orientation[:3], orientation[3:])))
    for key in ("SliceLocation", "InstanceNumber"):
        if hasattr(ds, key):
            return float(getattr(ds, key))
    return None


def _list_dicom_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [str(path)]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    files = [
        os.path.join(root, name)
        for root, _, names in os.walk(path)
        for name in names
        if not name.startswith(".")
        and (name.lower().endswith((".dcm", ".dicom")) or "." not in name)
    ]
    if not files:
        raise FileNotFoundError(f"No DICOM-like files found under {path}")
    return sorted(files)


def _orientation(ds) -> tuple[np.ndarray, np.ndarray] | None:
    has_position = hasattr(ds, "ImagePositionPatient")
    has_orientation = hasattr(ds, "ImageOrientationPatient")
    if has_position != has_orientation:
        raise ValueError("DICOM must supply position and orientation together")
    if not has_position:
        return None
    orient = np.asarray(ds.ImageOrientationPatient, dtype=float)
    position = np.asarray(ds.ImagePositionPatient, dtype=float)
    if (
        orient.shape != (6,)
        or position.shape != (3,)
        or not np.isfinite(orient).all()
        or not np.isfinite(position).all()
    ):
        raise ValueError(
            "DICOM orientation/position must contain finite direction cosines/coordinates"
        )
    row, col = orient[:3], orient[3:]
    if not (
        np.isclose(np.linalg.norm(row), 1, atol=1e-5)
        and np.isclose(np.linalg.norm(col), 1, atol=1e-5)
        and np.isclose(np.dot(row, col), 0, atol=1e-5)
    ):
        raise ValueError("DICOM orientation must contain orthogonal unit vectors")
    return orient, position


def load_dicom_series(
    path: str, *, strict: bool = True, include_patient_id: bool = False
) -> MedVolume:
    """Load supported classic monochrome DICOM images on a regular grid.

    Full inter-slice vectors preserve regular gantry tilt/shear. The affine maps
    array indices to RAS mm. MONOCHROME1 values are retained in modality units;
    photometric interpretation is recorded for the display layer. Enhanced and
    multiframe data and modality LUT sequences are explicitly unsupported.

    Args:
        path: Directory or single DICOM filename.
        strict: Fail on unreadable/nonimage candidates and missing stack geometry.
            False permits skipped files and unknown geometry with warnings and
            diagnostics in metadata. Structural inconsistencies always raise.
        include_patient_id: Opt in to retaining PatientID; never anonymizes pixels.
    """
    pydicom = _import_pydicom()
    datasets = []
    skipped = []
    for filename in _list_dicom_files(path):
        try:
            ds = pydicom.dcmread(filename)
            if not hasattr(ds, "PixelData"):
                raise ValueError("no PixelData")
        except Exception as exc:
            if strict:
                raise ValueError(
                    f"Unreadable DICOM candidate {filename}: {type(exc).__name__}"
                ) from exc
            skipped.append(filename)
            continue
        if (
            int(_safe_get(ds, "NumberOfFrames", 1)) != 1
            or hasattr(ds, "PerFrameFunctionalGroupsSequence")
            or hasattr(ds, "SharedFunctionalGroupsSequence")
        ):
            raise ValueError(
                "Enhanced/multiframe DICOM is unsupported; use a validated per-frame converter"
            )
        if str(_safe_get(ds, "SOPClassUID", "")) not in SUPPORTED_SOP_CLASSES:
            raise ValueError(
                "Unsupported DICOM SOP class; only classic CT/MR/CR/DX/MG are supported"
            )
        if int(_safe_get(ds, "SamplesPerPixel", 1)) != 1 or str(
            ds.PhotometricInterpretation
        ) not in ("MONOCHROME1", "MONOCHROME2"):
            raise ValueError("Only monochrome DICOM images are supported")
        if hasattr(ds, "ModalityLUTSequence") or hasattr(ds, "RealWorldValueMappingSequence"):
            raise ValueError("DICOM LUT/real-world mapping sequences require a validated converter")
        try:
            pixels = ds.pixel_array
        except Exception as exc:
            if strict:
                raise ValueError(f"Cannot decode DICOM pixel data: {filename}") from exc
            skipped.append(filename)
            continue
        if pixels.ndim != 2:
            raise ValueError(f"Expected 2D slices, got shape {pixels.shape}")
        datasets.append((ds, pixels))
    if not datasets:
        raise FileNotFoundError(f"No readable DICOM pixel data under {path}")
    if skipped:
        warnings.warn(
            f"Skipped {len(skipped)} unreadable/nonimage DICOM candidates",
            UserWarning,
            stacklevel=2,
        )
    for key in ("SeriesInstanceUID", "StudyInstanceUID", "FrameOfReferenceUID", "SOPClassUID"):
        values = [str(_safe_get(ds, key, "")) for ds, _ in datasets]
        present = {v for v in values if v}
        if len(present) > 1 or (present and any(not v for v in values)):
            label = "multiple DICOM series or " if key == "SeriesInstanceUID" else ""
            raise ValueError(f"{label}inconsistent {key} across DICOM slices")
        if key == "SeriesInstanceUID" and strict and not present:
            raise ValueError("DICOM SeriesInstanceUID is required in strict mode")
    geometries = [_orientation(ds) for ds, _ in datasets]
    known = all(g is not None for g in geometries)
    if any(g is not None for g in geometries) and not known:
        raise ValueError("DICOM slices have inconsistent geometry availability")
    ref = datasets[0][0]
    pixel_spacing = np.asarray(_safe_get(ref, "PixelSpacing", [1, 1]), dtype=float)
    if (
        pixel_spacing.shape != (2,)
        or not np.isfinite(pixel_spacing).all()
        or (pixel_spacing <= 0).any()
    ):
        raise ValueError("DICOM PixelSpacing must be finite and positive")
    for ds, pixels in datasets:
        spacing = np.asarray(_safe_get(ds, "PixelSpacing", [1, 1]), dtype=float)
        if (
            pixels.shape != datasets[0][1].shape
            or spacing.shape != (2,)
            or not np.allclose(spacing, pixel_spacing)
        ):
            raise ValueError("DICOM slices have inconsistent shape or PixelSpacing")
    if known:
        if any(not hasattr(ds, "PixelSpacing") for ds, _ in datasets):
            raise ValueError("Patient-space DICOM geometry requires PixelSpacing")
        first_geometry = geometries[0]
        assert first_geometry is not None
        orientation = first_geometry[0]
        if any(
            g is None or not np.allclose(g[0], orientation, atol=1e-5, rtol=0) for g in geometries
        ):
            raise ValueError("DICOM slices have inconsistent orientation")
        normal = np.cross(orientation[:3], orientation[3:])
        datasets.sort(key=lambda item: float(np.dot(item[0].ImagePositionPatient, normal)))
    elif len(datasets) > 1 and strict:
        raise ValueError(
            "DICOM stack patient geometry is missing; use strict=False for an unknown grid"
        )
    else:
        datasets.sort(key=lambda item: _slice_position(item[0]) or 0)
        warnings.warn("DICOM patient geometry is unknown", UserWarning, stacklevel=2)
    ref = datasets[0][0]
    affine = None
    single = len(datasets) == 1
    z_spacing = abs(
        float(_safe_get(ref, "SpacingBetweenSlices", _safe_get(ref, "SliceThickness", 1)))
    )
    if known:
        positions = np.asarray([ds.ImagePositionPatient for ds, _ in datasets], dtype=float)
        if not single:
            steps = np.diff(positions, axis=0)
            projected = steps @ normal
            if (projected <= 1e-6).any():
                raise ValueError("DICOM slice positions must be distinct")
            step = np.median(steps, axis=0)
            if not np.allclose(steps, step, rtol=1e-3, atol=1e-3):
                raise ValueError("DICOM slice spacing/vectors are irregular; resample first")
            z_spacing = float(np.linalg.norm(step))
        else:
            step = normal * z_spacing
        affine = np.eye(4)
        lps_to_ras = np.diag([-1.0, -1.0, 1.0])
        # IOP first vector follows columns, second follows rows.
        basis = (
            np.column_stack(
                [orientation[3:] * pixel_spacing[0], orientation[:3] * pixel_spacing[1], normal]
            )
            if single
            else np.column_stack(
                [step, orientation[3:] * pixel_spacing[0], orientation[:3] * pixel_spacing[1]]
            )
        )
        affine[:3, :3] = lps_to_ras @ basis
        affine[:3, 3] = lps_to_ras @ positions[0]
    slices = []
    for ds, pixels in datasets:
        slope = float(_safe_get(ds, "RescaleSlope", 1))
        intercept = float(_safe_get(ds, "RescaleIntercept", 0))
        if not np.isfinite([slope, intercept]).all() or slope == 0:
            raise ValueError("DICOM rescale slope/intercept must be finite; slope must be nonzero")
        slices.append(pixels.astype(np.float32) * slope + intercept)
    image = slices[0] if single else np.stack(slices)
    metadata = _build_metadata(ref, source=path, include_patient_id=include_patient_id)
    metadata.update(num_slices=len(slices), skipped_files=skipped, geometry_known=known)
    output_spacing = tuple(pixel_spacing) if single else (z_spacing, *pixel_spacing)
    return MedVolume(image=image, spacing=output_spacing, metadata=metadata, affine=affine)


def _build_metadata(ds, *, source: str, include_patient_id: bool = False) -> dict:
    metadata = {
        "source": str(source),
        "modality": str(_safe_get(ds, "Modality", "")) or None,
        "vendor": str(_safe_get(ds, "Manufacturer", "")) or None,
        "study_uid": str(_safe_get(ds, "StudyInstanceUID", "")) or None,
        "series_uid": str(_safe_get(ds, "SeriesInstanceUID", "")) or None,
        "photometric_interpretation": str(ds.PhotometricInterpretation),
        "rescale_slope": float(_safe_get(ds, "RescaleSlope", 1)),
        "rescale_intercept": float(_safe_get(ds, "RescaleIntercept", 0)),
    }
    if include_patient_id:
        metadata["patient_id"] = str(_safe_get(ds, "PatientID", "")) or None
    for tag, key in (("WindowCenter", "window_center"), ("WindowWidth", "window_width")):
        value = _safe_get(ds, tag)
        if value is not None:
            metadata[key] = float(
                value[0] if hasattr(value, "__len__") and not isinstance(value, str) else value
            )
    return metadata
