"""Multiplicative bias field — MRI intensity non-uniformity simulation."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import zoom

from medaugment.core.base import Transform
from medaugment.core.utils import SeedLike, as_float32
from medaugment.core.volume import MedVolume


class BiasField(Transform):
    """Smooth multiplicative bias field simulating MRI intensity non-uniformity.

    MRI scans exhibit spatially-varying signal because RF coil sensitivity,
    B0 field inhomogeneity, and other hardware effects modulate the apparent
    tissue intensity across the field of view.  This transform approximates
    that effect by generating a smooth random log-scale field on a coarse
    grid, upsampling it to the full volume size, and multiplying the image
    pointwise.

    The multiplicative field is ``exp(F)`` where ``F`` is drawn from
    ``U(-alpha, alpha)`` on a coarse grid of shape ``coarse_shape``, then
    spline-upsampled.  Effective per-voxel multiplier range is
    ``[exp(-alpha), exp(alpha)]``.

    For typical MRI bias-field augmentation use ``alpha=0.3`` (±30% signal
    variation).  Larger values (e.g. ``0.5``) model strong coil effects.

    Args:
        alpha: Maximum log-scale deviation.  Must be >= 0.
        coarse_shape: Shape of the coarse random grid.  An integer means the
            same size for every axis; pass a tuple for per-axis control.
        order: Spline interpolation order for upsampling (3 = cubic).
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        coarse_shape: int | tuple[int, ...] = 4,
        order: int = 3,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.alpha = float(alpha)
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.coarse_shape = coarse_shape
        if order not in (0, 1, 2, 3):
            raise ValueError("order must be 0, 1, 2, or 3")
        self.order = int(order)

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)
        shape = image.shape
        ndim = len(shape)

        if isinstance(self.coarse_shape, int):
            coarse: tuple[int, ...] = tuple(self.coarse_shape for _ in range(ndim))
        else:
            coarse = tuple(int(c) for c in self.coarse_shape)
            if len(coarse) != ndim:
                raise ValueError(
                    f"coarse_shape has {len(coarse)} dims but volume has {ndim}"
                )

        if self.alpha == 0.0:
            return volume

        log_field_coarse = self.rng.uniform(
            -self.alpha, self.alpha, size=coarse
        ).astype(np.float32)

        zoom_factors = tuple(s / c for s, c in zip(shape, coarse))
        log_field = zoom(
            log_field_coarse, zoom_factors, order=self.order, mode="nearest"
        ).astype(np.float32)

        # Trim to exact shape (zoom may overshoot by 1 voxel due to rounding)
        slices = tuple(slice(0, s) for s in shape)
        log_field = log_field[slices]
        if log_field.shape != shape:
            pad = [(0, max(0, s - f)) for s, f in zip(shape, log_field.shape)]
            log_field = np.pad(log_field, pad, mode="edge")[slices]

        out = (image * np.exp(log_field)).astype(np.float32)
        return volume.replace(image=out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {
                "alpha": self.alpha,
                "coarse_shape": (
                    self.coarse_shape
                    if isinstance(self.coarse_shape, int)
                    else list(self.coarse_shape)
                ),
                "order": self.order,
                "p": self.p,
            },
        }
