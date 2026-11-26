"""Slice extraction, display windowing and dependency-free PNG encoding."""

from __future__ import annotations

import base64
import math
import struct
import zlib
from typing import Any

import numpy as np

from medaugmentx.core.volume import MedVolume

AXIS_NAMES_3D = ("z", "y", "x")
AXIS_NAMES_2D = ("y", "x")

# Distinct, colour-blind-aware label colours (Okabe-Ito, then Tableau extras).
LABEL_COLOURS = np.array(
    [
        (230, 159, 0),
        (86, 180, 233),
        (0, 158, 115),
        (240, 228, 66),
        (0, 114, 178),
        (213, 94, 0),
        (204, 121, 167),
        (148, 103, 189),
        (140, 86, 75),
        (188, 189, 34),
        (23, 190, 207),
        (227, 119, 194),
    ],
    dtype=np.float32,
)


def encode_png(pixels: np.ndarray) -> bytes:
    """Encode a uint8 ``(H, W)`` grayscale or ``(H, W, 3)`` RGB array as PNG."""
    if pixels.dtype != np.uint8 or pixels.ndim not in (2, 3):
        raise ValueError("encode_png expects a 2D grayscale or 3-channel uint8 array")
    if pixels.ndim == 3 and pixels.shape[2] != 3:
        raise ValueError("encode_png expects RGB data with three channels")
    height, width = pixels.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Cannot encode an empty image")
    colour_type = 0 if pixels.ndim == 2 else 2
    rows = np.ascontiguousarray(pixels).reshape(height, -1)
    raw = np.concatenate([np.zeros((height, 1), dtype=np.uint8), rows], axis=1).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def png_data_url(pixels: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(pixels)).decode("ascii")


def axis_names(ndim: int) -> tuple[str, ...]:
    return AXIS_NAMES_3D if ndim == 3 else AXIS_NAMES_2D


def slice_index(length: int, position: float) -> int:
    """Index for a relative position in ``[0, 1]`` along an axis of ``length``."""
    position = min(max(float(position), 0.0), 1.0)
    # Round half up so the centre of an even axis matches ``length // 2``.
    return int(math.floor(position * (length - 1) + 0.5))


def take_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    return array if array.ndim == 2 else np.take(array, index, axis=axis)


def display_window(array: np.ndarray) -> tuple[float, float]:
    """Robust 1st-99th percentile window of the finite values."""
    finite = array[np.isfinite(array)]
    if not finite.size:
        return 0.0, 1.0
    low, high = (float(v) for v in np.percentile(finite, (1, 99)))
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        high = low + 1.0
    return low, high


def to_uint8(image: np.ndarray, low: float, high: float) -> np.ndarray:
    scaled = (np.nan_to_num(image.astype(np.float64), nan=low) - low) / (high - low)
    return (np.clip(scaled, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def overlay_labels(gray: np.ndarray, labels: np.ndarray | None, alpha: float) -> np.ndarray:
    """Blend label colours over a grayscale slice; returns RGB uint8."""
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    if labels is not None and alpha > 0:
        present = labels != 0
        if present.any():
            ids = np.asarray(labels[present])
            count = len(LABEL_COLOURS)
            if ids.dtype.kind == "b":
                index = np.zeros(ids.shape, dtype=np.int64)
            else:
                # Remainder in the label dtype: huge uint64 IDs cannot overflow.
                residue = np.remainder(ids, count).astype(np.int64)
                index = (residue - 1) % count
            rgb[present] = (1 - alpha) * rgb[present] + alpha * LABEL_COLOURS[index]
    return (rgb + 0.5).astype(np.uint8)


def difference_image(before: np.ndarray, after: np.ndarray) -> np.ndarray | None:
    """Absolute difference on a shared window, or None when grids differ."""
    if before.shape != after.shape:
        return None
    delta = np.abs(after.astype(np.float64) - before.astype(np.float64))
    finite = delta[np.isfinite(delta)]
    high = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    return to_uint8(delta, 0.0, high if high > 0 else 1.0)


def project_targets(volume: MedVolume, axis: int, index: int) -> dict[str, Any]:
    """Keypoints and box cross-sections lying on the displayed slice.

    Returned coordinates are ``(row, column)`` in the 2D slice. Keypoints are
    kept within half a voxel of the slice; boxes are kept when they span it.
    """
    keep = [a for a in range(volume.ndim) if volume.ndim == 2 or a != axis]
    points: list[dict[str, Any]] = []
    boxes: list[dict[str, Any]] = []
    if volume.keypoints is not None:
        for i, point in enumerate(volume.keypoints):
            if volume.ndim == 3 and abs(float(point[axis]) - index) > 0.5:
                continue
            label = None if volume.keypoint_labels is None else volume.keypoint_labels[i]
            points.append(
                {
                    "row": float(point[keep[0]]),
                    "col": float(point[keep[1]]),
                    "label": None if label is None else str(label),
                }
            )
    if volume.bboxes is not None:
        ndim = volume.ndim
        for i, box in enumerate(volume.bboxes):
            if ndim == 3 and not float(box[axis]) - 0.5 <= index <= float(box[ndim + axis]) + 0.5:
                continue
            label = None if volume.bbox_labels is None else volume.bbox_labels[i]
            boxes.append(
                {
                    "row0": float(box[keep[0]]),
                    "col0": float(box[keep[1]]),
                    "row1": float(box[ndim + keep[0]]),
                    "col1": float(box[ndim + keep[1]]),
                    "label": None if label is None else str(label),
                }
            )
    return {"keypoints": points, "boxes": boxes}


def pixel_aspect(volume: MedVolume, axis: int) -> float:
    """Displayed height/width ratio of one pixel on the chosen slice plane."""
    keep = [a for a in range(volume.ndim) if volume.ndim == 2 or a != axis]
    return float(volume.spacing[keep[0]] / volume.spacing[keep[1]])


def render_view(
    volume: MedVolume,
    *,
    axis: int,
    position: float,
    channel: str | None,
    window: tuple[float, float],
    mask_alpha: float,
) -> dict[str, Any]:
    """Render one slice of a volume with its overlays as a JSON-ready dict."""
    image = volume.image if channel is None else volume.channels[channel]
    index = slice_index(volume.shape[axis], position) if volume.ndim == 3 else 0
    plane = take_slice(image, axis, index)
    labels = None if volume.mask is None else take_slice(volume.mask, axis, index)
    gray = to_uint8(plane, *window)
    return {
        "png": png_data_url(overlay_labels(gray, labels, mask_alpha)),
        "plain": plane,
        "index": index,
        "length": int(volume.shape[axis]) if volume.ndim == 3 else 1,
        "height": int(plane.shape[0]),
        "width": int(plane.shape[1]),
        "aspect": pixel_aspect(volume, axis),
        "targets": project_targets(volume, axis, index),
    }


__all__ = [
    "LABEL_COLOURS",
    "axis_names",
    "difference_image",
    "display_window",
    "encode_png",
    "overlay_labels",
    "pixel_aspect",
    "png_data_url",
    "project_targets",
    "render_view",
    "slice_index",
    "take_slice",
    "to_uint8",
]
