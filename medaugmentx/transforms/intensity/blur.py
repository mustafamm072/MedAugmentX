"""Spatial blur and resolution-degradation transforms."""
from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, zoom

from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike, as_float32
from medaugmentx.core.volume import MedVolume

Range = Union[float, tuple[float, float]]
IntRange = Union[int, tuple[int, int]]


class GaussianBlur(Transform):
    """Isotropic Gaussian blur applied to the image.

    The mask is **never** blurred — blurring annotation boundaries degrades
    segmentation labels.  For anisotropic sigma control call
    ``scipy.ndimage.gaussian_filter`` directly.

    Typical values:

    * 2-D mammography / DXR: ``sigma=(0.5, 1.5)``
    * 3-D MRI / CT: ``sigma=(0.5, 2.0)``

    Args:
        sigma: Blur sigma in voxels.  Scalar → fixed; tuple ``(lo, hi)``
            → sampled per call.  Same sigma applied to every axis.
        order: Gaussian derivative order (0 = standard blur).
        mode: Boundary mode for :func:`scipy.ndimage.gaussian_filter`.
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        sigma: Range = (0.5, 1.5),
        order: int = 0,
        mode: str = "reflect",
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if isinstance(sigma, (int, float)):
            v = float(sigma)
            if v < 0:
                raise ValueError("sigma must be >= 0")
            self.sigma_range: tuple[float, float] = (v, v)
        else:
            lo, hi = float(sigma[0]), float(sigma[1])
            if lo < 0 or hi < lo:
                raise ValueError(f"sigma range invalid: {sigma}")
            self.sigma_range = (lo, hi)
        if order not in (0, 1, 2, 3):
            raise ValueError("order must be 0, 1, 2, or 3")
        self.order = int(order)
        self.mode = mode

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)
        sigma = float(self.rng.uniform(*self.sigma_range))
        if sigma == 0.0:
            return volume
        blurred = gaussian_filter(image, sigma=sigma, order=self.order, mode=self.mode)
        return volume.replace(image=blurred.astype(np.float32, copy=False))

    def to_dict(self) -> dict[str, Any]:
        sr = self.sigma_range
        sigma: Any = sr[0] if sr[0] == sr[1] else list(sr)
        return {
            "name": self.__class__.__name__,
            "params": {
                "sigma": sigma,
                "order": self.order,
                "mode": self.mode,
                "p": self.p,
            },
        }


class SimulateLowResolution(Transform):
    """Simulate a lower-resolution acquisition by down- then upsampling.

    Downsamples the image by a random factor drawn from ``zoom_range``,
    then upsamples back to the original shape.  The resulting image contains
    the same number of voxels as the input but with blurred detail that
    mimics a scanner running at lower spatial resolution.

    This teaches models to be robust across scanner sites with different
    acquisition protocols — a common problem in multi-centre studies.

    The mask is left at full resolution (it represents the ground-truth
    annotation and remains valid regardless of image resolution).

    Args:
        zoom_range: ``(min, max)`` downsampling factor per axis; values
            in ``(0, 1)``.  E.g. ``(0.5, 0.9)`` means the temporary
            downsampled image has 50–90 % of the original voxels per side.
        order_down: Spline order for the downsampling step.
        order_up: Spline order for the upsampling step.
        per_axis: If True, sample an independent zoom factor per axis
            (useful for anisotropic volumes).  Default False.
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        zoom_range: tuple[float, float] = (0.5, 0.9),
        order_down: int = 1,
        order_up: int = 1,
        per_axis: bool = False,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        lo, hi = float(zoom_range[0]), float(zoom_range[1])
        if not (0 < lo <= hi <= 1.0):
            raise ValueError(f"zoom_range must satisfy 0 < lo <= hi <= 1, got {zoom_range}")
        self.zoom_range: tuple[float, float] = (lo, hi)
        if order_down not in (0, 1, 2, 3):
            raise ValueError("order_down must be 0–3")
        if order_up not in (0, 1, 2, 3):
            raise ValueError("order_up must be 0–3")
        self.order_down = int(order_down)
        self.order_up = int(order_up)
        self.per_axis = bool(per_axis)

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)
        ndim = volume.ndim
        original_shape = image.shape

        if self.per_axis:
            factors = [float(self.rng.uniform(*self.zoom_range)) for _ in range(ndim)]
        else:
            f = float(self.rng.uniform(*self.zoom_range))
            factors = [f] * ndim

        small = zoom(image, factors, order=self.order_down, mode="reflect")
        up_factors = [o / s for o, s in zip(original_shape, small.shape)]
        large = zoom(small, up_factors, order=self.order_up, mode="reflect")

        # Correct shape off-by-one from floating-point rounding
        slices = tuple(slice(0, s) for s in original_shape)
        large = large[slices]
        if large.shape != original_shape:
            pad = [(0, max(0, t - g)) for t, g in zip(original_shape, large.shape)]
            large = np.pad(large, pad, mode="edge")[slices]

        return volume.replace(image=large.astype(np.float32, copy=False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {
                "zoom_range": list(self.zoom_range),
                "order_down": self.order_down,
                "order_up": self.order_up,
                "per_axis": self.per_axis,
                "p": self.p,
            },
        }


class MedianBlur(Transform):
    """Median filter — edge-preserving smoothing, strong against salt-and-pepper.

    Unlike :class:`GaussianBlur`, the median filter removes impulse noise
    (hot/dead detector pixels, speckle) while keeping edges crisp.  The
    window is a cube of side ``ksize`` voxels applied across all axes.

    .. note::
        A 3-D median filter is a cubic-window rank filter and is intrinsically
        expensive on large volumes.  Prefer a small ``ksize`` (3) and/or a
        modest ``p`` on full-resolution 3-D data — see ``benchmarks/``.

    The mask is never modified.

    Args:
        ksize: Window side length in voxels.  Must be a **positive odd**
            integer, or a ``(low, high)`` inclusive range of odd integers
            sampled per call.
        mode: Boundary mode for :func:`scipy.ndimage.median_filter`.
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        ksize: IntRange = 3,
        mode: str = "reflect",
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if isinstance(ksize, int):
            lo = hi = int(ksize)
        else:
            lo, hi = int(ksize[0]), int(ksize[1])
        if lo < 1 or hi < lo:
            raise ValueError(f"ksize range invalid: {ksize}")
        if lo % 2 == 0 or hi % 2 == 0:
            raise ValueError(f"ksize bounds must be odd, got {ksize}")
        self.ksize_range: tuple[int, int] = (lo, hi)
        self.mode = mode

    def apply(self, volume: MedVolume) -> MedVolume:
        lo, hi = self.ksize_range
        # Sample among the odd values in [lo, hi].
        odd_choices = list(range(lo, hi + 1, 2))
        k = int(odd_choices[int(self.rng.integers(0, len(odd_choices)))])
        if k <= 1:
            return volume
        image = as_float32(volume.image)
        blurred = median_filter(image, size=k, mode=self.mode)
        return volume.replace(image=blurred.astype(np.float32, copy=False))

    def to_dict(self) -> dict[str, Any]:
        kr = self.ksize_range
        return {
            "name": self.__class__.__name__,
            "params": {
                "ksize": kr[0] if kr[0] == kr[1] else list(kr),
                "mode": self.mode,
                "p": self.p,
            },
        }
