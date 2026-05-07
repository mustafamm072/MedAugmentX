"""Elastic deformation with anisotropic alpha and sigma."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from medaugment.core.base import Transform
from medaugment.core.utils import SeedLike, as_float32
from medaugment.core.volume import MedVolume

ScalarOrSeq = Union[float, Sequence[float]]


def _broadcast(value: ScalarOrSeq, ndim: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return tuple(float(value) for _ in range(ndim))
    out = tuple(float(v) for v in value)
    if len(out) != ndim:
        raise ValueError(
            f"{name} must be a scalar or have {ndim} entries, got {len(out)}"
        )
    return out


class ElasticDeform(Transform):
    """B-spline-style elastic deformation via smoothed random displacements.

    Implementation follows the standard Simard / Castro-Pereira approach:

    1. Sample independent uniform displacement fields per spatial axis.
    2. Smooth each field with a Gaussian of axis-specific ``sigma``.
    3. Scale each field by axis-specific ``alpha``.
    4. Warp the image (linear interp) and mask (nearest neighbour) using
       :func:`scipy.ndimage.map_coordinates` with the same displacement.

    Anisotropic ``alpha`` and ``sigma`` are essential for tomosynthesis-like
    volumes where the slice axis has very different sampling than the
    in-plane axes (e.g. ``alpha=(120, 120, 10)``, ``sigma=(10, 10, 3)``).

    Args:
        alpha: Per-axis displacement magnitude in pixels.
        sigma: Per-axis Gaussian smoothing sigma.
        order: Spline order for image interpolation (mask always uses 0).
        mode: ``map_coordinates`` boundary mode.
        cval: Fill value for ``mode='constant'``.
        p: Probability of applying the transform.
        seed: RNG seed.
    """

    def __init__(
        self,
        alpha: ScalarOrSeq = 30.0,
        sigma: ScalarOrSeq = 4.0,
        order: int = 1,
        mode: str = "reflect",
        cval: float = 0.0,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.alpha_spec = alpha
        self.sigma_spec = sigma
        if order not in (0, 1, 2, 3):
            raise ValueError("order must be 0, 1, 2, or 3 for map_coordinates")
        self.order = int(order)
        self.mode = mode
        self.cval = float(cval)

    def _displacements(self, shape: tuple[int, ...]) -> list[np.ndarray]:
        ndim = len(shape)
        alphas = _broadcast(self.alpha_spec, ndim, "alpha")
        sigmas = _broadcast(self.sigma_spec, ndim, "sigma")
        fields = []
        for ax in range(ndim):
            raw = self.rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            if sigmas[ax] > 0:
                raw = gaussian_filter(raw, sigma=sigmas[ax], mode="reflect")
            fields.append(raw * float(alphas[ax]))
        return fields

    def apply(self, volume: MedVolume) -> MedVolume:
        shape = volume.shape
        displacements = self._displacements(shape)
        grid = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape], indexing="ij")
        coords = [g + d for g, d in zip(grid, displacements)]
        coord_stack = np.stack([c.ravel() for c in coords], axis=0)

        image = as_float32(volume.image)
        warped = map_coordinates(
            image,
            coord_stack,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
        ).reshape(shape)

        new_mask = None
        if volume.mask is not None:
            warped_mask = map_coordinates(
                volume.mask,
                coord_stack,
                order=0,
                mode="constant",
                cval=0,
            ).reshape(shape).astype(volume.mask.dtype, copy=False)
            new_mask = warped_mask
        return volume.replace(image=warped, mask=new_mask)
