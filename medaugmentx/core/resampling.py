"""Shared patient-grid mappings and categorical resampling."""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.ndimage import affine_transform

from medaugmentx.core.utils import sample_mask

GRID_TOLERANCE_MM = 1e-4


def _grid_corners(shape: tuple[int, ...]) -> np.ndarray:
    return np.array(list(product(*[(-0.5, n - 0.5) for n in shape])), dtype=np.float64)


def grids_match(first: np.ndarray, second: np.ndarray, shape: tuple[int, ...]) -> bool:
    """Compare physical positions across the sampling grid with an absolute tolerance."""
    ndim = len(shape)
    delta = first[:3] - second[:3]
    displacement = _grid_corners(shape) @ delta[:, :ndim].T + delta[:, 3]
    return bool(np.all(np.linalg.norm(displacement, axis=1) <= GRID_TOLERANCE_MM))


def grid_pull(
    source: np.ndarray, target: np.ndarray, shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the homogeneous grid map and its spatial sampling matrix/offset."""
    ndim = len(shape)
    basis = source[:3, :3]
    relative = target[:3].copy()
    pull = np.eye(4)
    with np.errstate(over="ignore", invalid="ignore"):
        relative[:, 3] -= source[:3, 3]
        pull[:3] = np.linalg.solve(basis, relative)
    if not np.isfinite(pull).all():
        raise ValueError("Source and target affines produce an unrepresentable grid mapping")
    if ndim == 3:
        return pull, pull[:3, :3], pull[:3, 3]

    # The inactive third affine column can point in any direction. Solve in
    # the plane itself so that column cannot affect the image sampling.
    plane = basis[:, :2]
    components = relative[:, [0, 1, 3]]
    mapped = np.linalg.lstsq(plane, components, rcond=None)[0]
    residual = components - plane @ mapped
    corner_residuals = _grid_corners(shape) @ residual[:, :2].T + residual[:, 2]
    if (
        np.any(np.linalg.norm(corner_residuals, axis=1) > GRID_TOLERANCE_MM)
        or np.linalg.matrix_rank(mapped[:, :2]) != 2
    ):
        raise ValueError("Source and target 2D patient-space grids must be coplanar")
    return pull, mapped[:, :2], mapped[:, 2]


def sample_grid_mask(
    mask: np.ndarray, matrix: np.ndarray, offset: np.ndarray, shape: tuple[int, ...]
) -> np.ndarray:
    """Sample categorical labels on a new grid using zero beyond the voxel extent."""
    return sample_mask(
        mask,
        lambda array: affine_transform(
            array,
            matrix,
            offset=offset,
            output_shape=shape,
            order=0,
            mode="grid-constant",
            cval=0,
            prefilter=False,
        ),
    )
