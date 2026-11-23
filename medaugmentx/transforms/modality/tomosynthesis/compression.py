"""CompressionVariation — DBT paddle-compression variation."""

from __future__ import annotations

import warnings
from typing import Any, Union

import numpy as np
from scipy.ndimage import affine_transform

from medaugmentx.core import geometry
from medaugmentx.core.base import Transform
from medaugmentx.core.parameters import validate_parameters
from medaugmentx.core.utils import SeedLike, as_float32, axis_label_to_index, sample_mask
from medaugmentx.core.volume import MedVolume

Range = Union[float, tuple[float, float]]
_SCIPY_1D_MATRIX = "The behavior of affine_transform with a 1-D array"


class CompressionVariation(Transform):
    """Simulate variation in breast-paddle compression for DBT.

    Between acquisitions the same breast is compressed by differing amounts,
    stretching or squeezing the tissue along the compression direction (the
    in-plane axis perpendicular to the chest wall).  This transform applies an
    anisotropic scaling along a single in-plane axis while keeping the volume
    shape and the slice axis untouched.

    The image and mask share the same sampled scale; the mask is resampled
    with nearest-neighbour interpolation so label values are preserved.
    Squeezing exposes background at both ends of the axis: the image is
    filled with ``0`` and the mask with label ``0``, for every ``order``.

    Args:
        scale: Multiplicative scale along the compression axis.  Scalar →
            fixed; tuple ``(lo, hi)`` → sampled per call.  Values < 1 squeeze,
            > 1 stretch.  Must be > 0.  Typical values ``0.85–1.15``.
        axis: In-plane compression axis — ``"y"`` (row, default) or ``"x"``.
        order: Image interpolation order (mask always uses ``0``).
        p: Probability of applying.
        seed: RNG seed.

    Raises:
        ValueError: If applied to a 2D volume — DBT is 3D by definition.
    """

    def __init__(
        self,
        scale: Range = (0.85, 1.15),
        axis: str = "y",
        order: int = 1,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        validate_parameters(scale=scale, axis=axis, order=order)
        if isinstance(scale, (int, float, np.integer, np.floating)):
            v = float(scale)
            if v <= 0:
                raise ValueError("scale must be > 0")
            self.scale_range: tuple[float, float] = (v, v)
        else:
            lo, hi = float(scale[0]), float(scale[1])
            if lo <= 0 or hi < lo:
                raise ValueError(f"scale range invalid: {scale}")
            self.scale_range = (lo, hi)
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        self.axis = axis
        if order not in (0, 1, 2, 3):
            raise ValueError("order must be 0, 1, 2, or 3")
        self.order = int(order)

    def _scale_axis_array(self, arr: np.ndarray, ax: int, s: float, order: int) -> np.ndarray:
        matrix = np.ones(arr.ndim, dtype=np.float64)
        matrix[ax] = 1.0 / s
        center = (arr.shape[ax] - 1) / 2.0
        offset = np.zeros(arr.ndim, dtype=np.float64)
        offset[ax] = center * (1.0 - 1.0 / s)
        mode = "constant"
        with warnings.catch_warnings():
            # SciPy < 1.14 warns about the 1-D (diagonal) matrix form on every
            # call. That form is intended: it keeps the exact zoom/shift path.
            warnings.filterwarnings("ignore", message=_SCIPY_1D_MATRIX, category=UserWarning)
            return affine_transform(
                arr,
                matrix=matrix,
                offset=offset,
                order=order,
                mode=mode,
                cval=0.0,
                output_shape=arr.shape,
            )

    def apply(self, volume: MedVolume) -> MedVolume:
        if not volume.is_3d:
            raise ValueError("CompressionVariation requires a 3D volume")
        ax = axis_label_to_index(self.axis, volume.ndim)
        s = float(self.rng.uniform(*self.scale_range))
        if abs(s - 1.0) < 1e-8:
            return volume

        new_mask = None
        if volume.mask is not None:
            new_mask = sample_mask(volume.mask, lambda arr: self._scale_axis_array(arr, ax, s, 0))

        forward = np.eye(volume.ndim)
        forward[ax, ax] = s
        centre = (np.asarray(volume.shape, dtype=np.float64) - 1.0) / 2.0
        return volume.warp(
            geometry.affine_map(forward, centre, np.zeros(volume.ndim)),
            image_map=lambda array: self._scale_axis_array(as_float32(array), ax, s, self.order),
            mask=new_mask,
        )

    def to_dict(self) -> dict[str, Any]:
        sr = self.scale_range
        return {
            "name": self.__class__.__name__,
            "params": {
                "scale": sr[0] if sr[0] == sr[1] else list(sr),
                "axis": self.axis,
                "order": self.order,
                "p": self.p,
                "seed": self._seed,
            },
        }
