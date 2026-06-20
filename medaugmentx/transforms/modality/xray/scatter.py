"""X-ray scatter (veiling glare) simulation."""
from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy.ndimage import gaussian_filter

from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike, as_float32
from medaugmentx.core.volume import MedVolume

Range = Union[float, tuple[float, float]]


class ScatterSimulation(Transform):
    """Simulate X-ray scatter (veiling glare) that lowers contrast.

    Compton-scattered photons add a smooth, low-frequency haze across the
    detector, reducing local contrast — especially pronounced in thick body
    regions and when no anti-scatter grid is used.  This transform adds a
    heavily blurred copy of the image as a scatter background.

    ``out = image + fraction * gaussian_blur(image, sigma)``

    The mask is never modified.

    Args:
        fraction: Scatter-to-primary amplitude ratio.  Scalar → fixed; tuple
            ``(lo, hi)`` → sampled per call.  Typical values ``0.1–0.5``.
        sigma: Gaussian sigma (voxels) of the scatter blur — large, since
            scatter is low-frequency.  Scalar → fixed; tuple → sampled.
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        fraction: Range = (0.1, 0.4),
        sigma: Range = (15.0, 40.0),
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.fraction_range = _as_range(fraction, "fraction", min_value=0.0)
        self.sigma_range = _as_range(sigma, "sigma", min_value=0.0)
        if self.sigma_range[0] <= 0:
            raise ValueError("sigma must be > 0")

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)
        fraction = float(self.rng.uniform(*self.fraction_range))
        sigma = float(self.rng.uniform(*self.sigma_range))
        if fraction == 0.0:
            return volume
        scatter = gaussian_filter(image, sigma=sigma, mode="reflect")
        out = image + fraction * scatter
        return volume.replace(image=out.astype(np.float32, copy=False))

    def to_dict(self) -> dict[str, Any]:
        fr, sr = self.fraction_range, self.sigma_range
        return {
            "name": self.__class__.__name__,
            "params": {
                "fraction": fr[0] if fr[0] == fr[1] else list(fr),
                "sigma": sr[0] if sr[0] == sr[1] else list(sr),
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
