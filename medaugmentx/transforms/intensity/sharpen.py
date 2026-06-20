"""Unsharp-mask sharpening."""
from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy.ndimage import gaussian_filter

from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike, as_float32
from medaugmentx.core.volume import MedVolume

Range = Union[float, tuple[float, float]]


class Sharpen(Transform):
    """Sharpen the image with an unsharp mask.

    Computes a blurred copy and adds back the high-frequency residual:

    ``out = image + alpha * (image - gaussian_blur(image, sigma))``

    Mimics the edge-enhancement filters radiologists routinely apply at the
    viewing workstation, and the variation between different vendor recon
    kernels.  The mask is never modified.

    Args:
        alpha: Sharpening strength.  Scalar → fixed; tuple ``(lo, hi)`` →
            sampled per call.  ``0`` is identity; typical values ``0.2–1.5``.
        sigma: Gaussian sigma (voxels) of the blur used to extract detail.
            Scalar → fixed; tuple → sampled per call.
        clip: Optional ``(min, max)`` to clamp the result.
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        alpha: Range = (0.2, 0.8),
        sigma: Range = (0.7, 1.5),
        clip: tuple[float, float] | None = None,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.alpha_range = _as_range(alpha, "alpha", min_value=0.0)
        self.sigma_range = _as_range(sigma, "sigma", min_value=0.0)
        if self.sigma_range[0] <= 0:
            raise ValueError("sigma must be > 0")
        self.clip = None if clip is None else (float(clip[0]), float(clip[1]))

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)
        alpha = float(self.rng.uniform(*self.alpha_range))
        sigma = float(self.rng.uniform(*self.sigma_range))
        if alpha == 0.0:
            return volume
        blurred = gaussian_filter(image, sigma=sigma, mode="reflect")
        out = image + alpha * (image - blurred)
        if self.clip is not None:
            out = np.clip(out, self.clip[0], self.clip[1])
        return volume.replace(image=out.astype(np.float32, copy=False))

    def to_dict(self) -> dict[str, Any]:
        ar, sr = self.alpha_range, self.sigma_range
        return {
            "name": self.__class__.__name__,
            "params": {
                "alpha": ar[0] if ar[0] == ar[1] else list(ar),
                "sigma": sr[0] if sr[0] == sr[1] else list(sr),
                "clip": list(self.clip) if self.clip is not None else None,
                "p": self.p,
            },
        }


def _as_range(x: Range, name: str, min_value: float) -> tuple[float, float]:
    if isinstance(x, (int, float)):
        v = float(x)
        if v < min_value:
            raise ValueError(f"{name} must be >= {min_value}, got {v}")
        return (v, v)
    lo, hi = float(x[0]), float(x[1])
    if lo < min_value or hi < lo:
        raise ValueError(f"{name} range invalid: {x}")
    return lo, hi
