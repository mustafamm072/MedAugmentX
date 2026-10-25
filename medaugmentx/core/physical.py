"""Patient-space geometry: array indices to RAS coordinates in millimetres."""

from __future__ import annotations

import numpy as np


def axis_permutation(ndim: int) -> np.ndarray:
    """Map array coordinates (zyx/yx) to file coordinates (xyz/xy)."""
    out = np.eye(4)
    out[:ndim, :ndim] = np.eye(ndim)[::-1]
    return out


def validate_affine(value: object) -> np.ndarray:
    """Return an owned, finite, nonsingular homogeneous 4x4 affine."""
    affine = np.array(value, dtype=np.float64, copy=True)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("affine must be a finite 4x4 matrix")
    if not np.allclose(affine[3], [0, 0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError("affine must have homogeneous last row [0, 0, 0, 1]")
    if np.linalg.matrix_rank(affine[:3, :3]) != 3:
        raise ValueError("affine spatial matrix must be nonsingular")
    return affine


def affine_spacing(affine: np.ndarray, ndim: int) -> tuple[float, ...]:
    """Lengths of the spatial basis vectors, including sheared grids."""
    return tuple(float(x) for x in np.linalg.norm(affine[:3, :ndim], axis=0))


def grid_translation(offset: object) -> np.ndarray:
    """Homogeneous output-to-input translation in array-axis order."""
    vector = np.asarray(offset, dtype=float)
    out = np.eye(4)
    out[: len(vector), 3] = vector
    return out


def assumed_affine(spacing: tuple[float, ...]) -> np.ndarray:
    """Axis-aligned zero-origin geometry for explicitly assumed-grid export."""
    native = np.eye(4)
    for axis, value in enumerate(reversed(spacing)):
        native[axis, axis] = value
    return native @ axis_permutation(len(spacing))
