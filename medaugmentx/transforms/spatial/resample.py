"""Resampling onto an explicitly spaced, centre-aligned voxel grid."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any, cast

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter

from medaugmentx.core import geometry
from medaugmentx.core.base import Transform
from medaugmentx.core.parameters import positive_size, validate_parameters
from medaugmentx.core.utils import SeedLike, as_float32
from medaugmentx.core.volume import MedVolume


class ResampleToSpacing(Transform):
    """Resample to a requested voxel spacing in millimetres along the grid axes.

    Keeps the physical image centre and grid directions (including shear).
    Each output size is ``max(1, floor(size * old_spacing / spacing + 0.5))``.
    Spacing is exact; rounding changes the voxel extent by at most half an
    output voxel per axis, except when the minimum size of one is needed.
    Unlike Resize, this does not preserve the exact field of view or produce
    a fixed batch shape. Border samples use nearest-edge extension.

    Args:
        spacing: Positive finite millimetres, either an isotropic scalar or
            one entry per array axis (z, y, x for 3D; y, x for 2D).
        order: Image spline interpolation order (0–3); masks always use 0.
        p: Probability of applying; use 1 for deterministic preprocessing.
        seed: RNG seed, used only for probability gating.
        antialias: Smooth the image before downsampling; never filters masks.
        clip_output: Clamp spline overshoot to the input intensity range.
        max_voxels: Optional positive output voxel limit, checked before
            allocating resampled arrays. None disables this limit.
    """

    def __init__(
        self,
        spacing: float | Sequence[float],
        order: int = 1,
        p: float = 1.0,
        seed: SeedLike = None,
        antialias: bool = True,
        clip_output: bool = False,
        max_voxels: int | None = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        values = (spacing,) if isinstance(spacing, Real) else tuple(cast(Sequence[float], spacing))
        if not values or any(
            isinstance(v, (bool, np.bool_))
            or not isinstance(v, Real)
            or not math.isfinite(v)
            or v <= 0
            for v in values
        ):
            raise ValueError("spacing must contain positive finite numbers")
        self.spacing = (
            float(spacing) if isinstance(spacing, Real) else tuple(float(v) for v in values)
        )
        validate_parameters(order=order)
        if order not in (0, 1, 2, 3):
            raise ValueError("order must be 0, 1, 2, or 3")
        if max_voxels is not None:
            positive_size((max_voxels,), "max_voxels")
        self.order = int(order)
        self.antialias = bool(antialias)
        self.clip_output = bool(clip_output)
        self.max_voxels = None if max_voxels is None else int(max_voxels)

    def apply(self, volume: MedVolume) -> MedVolume:
        spacing = (self.spacing,) * volume.ndim if isinstance(self.spacing, float) else self.spacing
        if len(spacing) != volume.ndim:
            raise ValueError("spacing must have one entry per volume axis")
        # Compute sizes as Python integers: products must not wrap int64.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ratios = np.asarray(spacing) / np.asarray(volume.spacing)
            extents = np.asarray(volume.shape) / ratios
        if not np.isfinite(extents).all() or not np.isfinite(ratios).all() or (ratios <= 0).any():
            raise ValueError("Requested spacing produces an unrepresentable output grid")
        shape = tuple(max(1, math.floor(float(s) + 0.5)) for s in extents)
        if self.max_voxels is not None and math.prod(shape) > self.max_voxels:
            raise ValueError(f"Output shape {shape} exceeds max_voxels={self.max_voxels}")
        if math.prod(shape) > np.iinfo(np.intp).max // np.dtype(np.float32).itemsize:
            raise ValueError("Requested spacing produces an unrepresentable output grid")
        if tuple(spacing) == volume.spacing:
            return volume

        offset = (np.asarray(volume.shape) - 1 - ratios * (np.asarray(shape) - 1)) / 2
        image = as_float32(volume.image)
        if self.antialias and (ratios > 1).any():
            image = gaussian_filter(image, sigma=np.maximum(0, (ratios - 1) / 2))
        matrix = np.diag(ratios)
        new_image = affine_transform(
            image, matrix, offset=offset, output_shape=shape, order=self.order, mode="nearest"
        )
        if self.clip_output:
            new_image = np.clip(new_image, volume.image.min(), volume.image.max())
        new_mask = None
        if volume.mask is not None:
            # Direct nearest-neighbour indexing keeps even int64/uint64 IDs
            # exact; ndimage interpolation internally converts labels to doubles.
            indices = [
                np.clip(np.floor(ratio * np.arange(size) + shift + 0.5), 0, old - 1).astype(np.intp)
                for ratio, size, shift, old in zip(ratios, shape, offset, volume.shape)
            ]
            new_mask = volume.mask[np.ix_(*indices)]

        point_map = geometry.affine_map(
            np.diag(1 / ratios), np.zeros(volume.ndim), -offset / ratios
        )
        pull = np.eye(4)
        pull[: volume.ndim, : volume.ndim] = matrix
        pull[: volume.ndim, 3] = offset
        return volume.warp(
            point_map,
            image=new_image.astype(np.float32, copy=False),
            mask=new_mask,
            spacing=tuple(spacing),
            grid_map=pull,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": type(self).__name__,
            "params": {
                "spacing": self.spacing,
                "order": self.order,
                "p": self.p,
                "seed": self._seed,
                "antialias": self.antialias,
                "clip_output": self.clip_output,
                "max_voxels": self.max_voxels,
            },
        }
