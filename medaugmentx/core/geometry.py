"""Coordinate geometry for keypoint and bounding-box targets.

MedAugmentX carries two optional geometric targets on every
:class:`~medaugmentx.core.volume.MedVolume` — *keypoints* (landmark points) and
*bounding boxes* — so that detection and landmark annotations stay in lockstep
with the image and mask when spatial transforms warp the pixel grid.

Coordinate convention
---------------------
Both targets are stored in **array-index order**, matching the image axes:

- 3D volumes ``(D, H, W)`` → coordinates are ``(z, y, x)``.
- 2D images  ``(H, W)``    → coordinates are ``(y, x)``.

This is deliberately the same order used to index ``image``/``mask`` and to
address :func:`scipy.ndimage` displacement fields, so a spatial transform can
reuse the exact geometry it already computes for the pixels.  (Note this is the
transpose of the ``(x, y)`` order some 2D libraries use; convert on the way in
and out if you follow that convention.)

- ``keypoints``: ``float`` array of shape ``(N, ndim)``.
- ``bboxes``: ``float`` array of shape ``(M, 2 * ndim)`` laid out as
  ``[min_axis0, …, min_axis(ndim-1), max_axis0, …, max_axis(ndim-1)]`` —
  the low corner followed by the high corner, both in array-index order.

Every function here is a pure NumPy helper with no MedVolume dependency, so the
transforms can compose them freely.  A spatial transform only needs to express
its pixel mapping as a *point map* — a callable taking an ``(K, ndim)`` array of
coordinates and returning the transformed ``(K, ndim)`` array — and this module
threads that map through both keypoints and (via their corners) bounding boxes.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

PointMap = Callable[[np.ndarray], np.ndarray]


def as_keypoints(points: object, ndim: int) -> np.ndarray:
    """Validate and coerce ``points`` to a ``(N, ndim)`` float array."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != ndim:
        raise ValueError(
            f"keypoints must have shape (N, {ndim}) for a {ndim}D volume; got {arr.shape}"
        )
    return arr


def as_bboxes(boxes: object, ndim: int) -> np.ndarray:
    """Validate and coerce ``boxes`` to a ``(M, 2*ndim)`` float array.

    Enforces ``min <= max`` on every axis so a malformed box is caught at
    construction rather than silently producing a negative-extent region.
    """
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 * ndim:
        raise ValueError(
            f"bboxes must have shape (M, {2 * ndim}) for a {ndim}D volume; got {arr.shape}"
        )
    mins, maxs = arr[:, :ndim], arr[:, ndim:]
    if arr.size and (maxs < mins).any():
        raise ValueError("each bbox must satisfy min <= max on every axis")
    return arr


def map_keypoints(keypoints: np.ndarray, point_map: PointMap) -> np.ndarray:
    """Apply a coordinate ``point_map`` to a ``(N, ndim)`` keypoint array."""
    if keypoints.shape[0] == 0:
        return keypoints.copy()
    return np.asarray(point_map(keypoints), dtype=np.float64)


def _bbox_corners(bboxes: np.ndarray, ndim: int) -> np.ndarray:
    """Expand each axis-aligned box to its ``2**ndim`` corner coordinates.

    Returns an ``(M, 2**ndim, ndim)`` array.  A box is only guaranteed to stay
    axis-aligned under axis-aligned maps (flip, scale, translate); under a
    rotation the enclosing axis-aligned box of the transformed corners is the
    correct — and standard — conservative result.
    """
    mins, maxs = bboxes[:, :ndim], bboxes[:, ndim:]
    # Bit i of the corner index selects max (1) or min (0) for axis i.
    n_corners = 1 << ndim
    corners = np.empty((bboxes.shape[0], n_corners, ndim), dtype=np.float64)
    for c in range(n_corners):
        for ax in range(ndim):
            pick_max = (c >> ax) & 1
            corners[:, c, ax] = maxs[:, ax] if pick_max else mins[:, ax]
    return corners


def map_bboxes(bboxes: np.ndarray, ndim: int, point_map: PointMap) -> np.ndarray:
    """Map every box through ``point_map`` via its corners, re-bounding to AABB.

    Each box is expanded to its ``2**ndim`` corners, the corners are pushed
    through the same point map used for keypoints, and the transformed box is
    the axis-aligned bounding box of the mapped corners.  This keeps boxes
    correct under rotation and reflection, not only translation/scaling.
    """
    if bboxes.shape[0] == 0:
        return bboxes.copy()
    corners = _bbox_corners(bboxes, ndim)  # (M, 2**ndim, ndim)
    flat = corners.reshape(-1, ndim)
    mapped = np.asarray(point_map(flat), dtype=np.float64).reshape(corners.shape)
    mins = mapped.min(axis=1)
    maxs = mapped.max(axis=1)
    return np.concatenate([mins, maxs], axis=1)


# ---------------------------------------------------------------------------
# Point maps used by the spatial transforms
# ---------------------------------------------------------------------------


def flip_map(flip_axes: tuple[int, ...], shape: tuple[int, ...]) -> PointMap:
    """Point map reflecting coordinates along ``flip_axes`` of a grid ``shape``."""

    def _fn(pts: np.ndarray) -> np.ndarray:
        out = np.array(pts, dtype=np.float64, copy=True)
        for ax in flip_axes:
            out[:, ax] = (shape[ax] - 1) - out[:, ax]
        return out

    return _fn


def affine_map(
    forward: np.ndarray, centre: np.ndarray, translation_px: np.ndarray
) -> PointMap:
    """Point map for a forward affine warp about ``centre`` plus a translation.

    ``forward`` is the *forward* linear map (rotation @ scale); the image warp
    uses its inverse to pull pixels, so points move the opposite way — hence the
    forward matrix here.
    """

    def _fn(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        return (pts - centre) @ forward.T + centre + translation_px

    return _fn


def translate_map(offset: np.ndarray) -> PointMap:
    """Point map adding a constant ``offset`` (e.g. crop origin or pad width)."""

    def _fn(pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, dtype=np.float64) + offset

    return _fn


def scale_map(factors: np.ndarray) -> PointMap:
    """Point map scaling each axis independently (e.g. resample zoom factors)."""

    def _fn(pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, dtype=np.float64) * factors

    return _fn


def displacement_map(
    displacements: list[np.ndarray], mode: str = "reflect"
) -> PointMap:
    """Point map for an elastic displacement field.

    ``displacements[ax]`` is the per-axis field ``d`` such that the image warp
    reads ``out[x] = in[x + d[x]]``.  A landmark at input location ``p`` appears
    in the output near ``p - d(p)``; the field is sampled at each point with
    linear interpolation.  Exact for translations and a close approximation for
    the smooth fields elastic deformation produces.
    """
    from scipy.ndimage import map_coordinates

    def _fn(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape[0] == 0:
            return pts.copy()
        coords = pts.T  # (ndim, K)
        out = pts.copy()
        for ax, field in enumerate(displacements):
            sampled = map_coordinates(field, coords, order=1, mode=mode)
            out[:, ax] = pts[:, ax] - sampled
        return out

    return _fn


__all__ = [
    "PointMap",
    "as_keypoints",
    "as_bboxes",
    "map_keypoints",
    "map_bboxes",
    "flip_map",
    "affine_map",
    "translate_map",
    "scale_map",
    "displacement_map",
]
