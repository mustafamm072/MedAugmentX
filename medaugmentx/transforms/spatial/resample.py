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
from medaugmentx.core.physical import validate_affine
from medaugmentx.core.resampling import grid_pull, sample_grid_mask
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
        matrix = np.diag(ratios)

        def resample_image(array: np.ndarray) -> np.ndarray:
            image = as_float32(array)
            if self.antialias and (ratios > 1).any():
                image = gaussian_filter(image, sigma=np.maximum(0, (ratios - 1) / 2))
            out = affine_transform(
                image, matrix, offset=offset, output_shape=shape, order=self.order, mode="nearest"
            )
            if self.clip_output:
                out = np.clip(out, array.min(), array.max())
            return out.astype(np.float32, copy=False)

        new_mask = None
        if volume.mask is not None:
            # Sample axis index ramps with the same SciPy kernel as the image.
            # NumPy rounding can disagree at half-voxel boundaries. Indexing
            # the mask afterwards keeps int64/uint64 IDs out of floating point.
            indices = [
                affine_transform(
                    np.arange(old, dtype=np.float64),
                    np.array([[ratio]]),
                    offset=[shift],
                    output_shape=(size,),
                    order=0,
                    mode="nearest",
                ).astype(np.intp)
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
            image_map=resample_image,
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


class ResampleToGrid(Transform):
    """Resample already registered data onto an explicit patient-space grid.

    The target shape and array-index-to-RAS-mm affine determine the output
    field of view, orientation and spacing. For 2D, source and target planes
    must be coplanar. Uses float32 spline interpolation for the image and
    nearest-neighbour with zero padding for the mask. The image interpolates
    against ``cval`` beyond the source grid. There is no antialiasing filter,
    so downsampling can alias the image and omit small labels.

    Args:
        shape: Target shape in spatial array-axis order, with two or three entries.
        affine: Target 4x4 affine mapping array indices to RAS millimetres.
        order: Image spline order (0–3). The mask always uses nearest-neighbour.
        cval: Image fill in its current intensity units.
        max_voxels: Optional positive limit on the number of output voxels.
        p: Probability of applying. Use 1 for a fixed output grid.
        seed: RNG seed, used only for probability gating.
    """

    def __init__(
        self,
        shape: Sequence[int],
        affine: object,
        order: int = 1,
        *,
        cval: float = 0.0,
        max_voxels: int | None = None,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.shape = positive_size(shape, "shape")
        if len(self.shape) not in (2, 3):
            raise ValueError("shape must have two or three entries")
        self.affine = validate_affine(affine)
        self.affine.setflags(write=False)
        validate_parameters(order=order)
        if order not in (0, 1, 2, 3):
            raise ValueError("order must be 0, 1, 2, or 3")
        self.order = int(order)
        self.cval = float(cval)
        if not math.isfinite(self.cval) or abs(self.cval) > float(np.finfo(np.float32).max):
            raise ValueError("cval must be finite and representable as float32")
        if max_voxels is not None:
            positive_size((max_voxels,), "max_voxels")
        self.max_voxels = None if max_voxels is None else int(max_voxels)
        if self.max_voxels is not None and math.prod(self.shape) > self.max_voxels:
            raise ValueError(f"Output shape {self.shape} exceeds max_voxels={self.max_voxels}")
        if math.prod(self.shape) > np.iinfo(np.intp).max // np.dtype(np.float32).itemsize:
            raise ValueError("Requested shape produces an unrepresentable output grid")

    def apply(self, volume: MedVolume) -> MedVolume:
        if volume.affine is None:
            raise ValueError("Source patient geometry is unknown; supply an affine")
        if volume.ndim != len(self.shape):
            raise ValueError("Target grid and source volume must have the same ndim")
        if volume.shape == self.shape and np.array_equal(volume.affine, self.affine):
            return volume
        pull, matrix, offset = grid_pull(volume.affine, self.affine, self.shape)
        forward = np.linalg.inv(matrix)

        def resample_image(array: np.ndarray) -> np.ndarray:
            return affine_transform(
                as_float32(array),
                matrix,
                offset=offset,
                output_shape=self.shape,
                order=self.order,
                mode="grid-constant",
                cval=self.cval,
            )

        mask = (
            None
            if volume.mask is None
            else sample_grid_mask(volume.mask, matrix, offset, self.shape)
        )
        return volume.warp(
            geometry.affine_map(forward, np.zeros(volume.ndim), -(forward @ offset)),
            image_map=resample_image,
            mask=mask,
            grid_map=pull,
        )

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["params"]["affine"] = self.affine.tolist()
        return result
