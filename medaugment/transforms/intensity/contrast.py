"""Contrast & gamma transforms."""
from __future__ import annotations

from typing import Any, Union

import numpy as np

from medaugment.core.base import Transform
from medaugment.core.utils import SeedLike, as_float32
from medaugment.core.volume import MedVolume

Range = Union[float, tuple[float, float]]


class GammaCorrection(Transform):
    """Power-law (gamma) intensity transformation.

    Applied on the **normalised** image (rescaled to ``[0, 1]`` per call),
    then unscaled back to the original intensity range. This makes the
    transform behave consistently regardless of whether the input is
    ``[0, 1]``, raw HU, or scanner-native MR intensities.

    For breast tomosynthesis a common range is ``gamma=(0.85, 1.15)``.

    Args:
        gamma: Either a fixed gamma or a ``(low, high)`` range. Values < 1
            brighten mid-tones, values > 1 darken mid-tones.
        invert: If True, apply gamma to the inverted image (useful for
            contrast on dark backgrounds).
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        gamma: Range = (0.8, 1.2),
        invert: bool = False,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if isinstance(gamma, (int, float)):
            self.gamma_range: tuple[float, float] = (float(gamma), float(gamma))
        else:
            lo, hi = float(gamma[0]), float(gamma[1])
            if lo <= 0 or hi < lo:
                raise ValueError(f"gamma range must be positive and ordered, got {gamma}")
            self.gamma_range = (lo, hi)
        self.invert = bool(invert)

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)
        g = float(self.rng.uniform(*self.gamma_range))

        lo = float(image.min())
        hi = float(image.max())
        if hi - lo < 1e-12:
            return volume  # constant image — gamma is identity

        norm = (image - lo) / (hi - lo)
        if self.invert:
            norm = 1.0 - norm
        powered = np.power(norm, g, dtype=np.float32)
        if self.invert:
            powered = 1.0 - powered
        out = powered * (hi - lo) + lo
        return volume.replace(image=out.astype(np.float32, copy=False))

    def to_dict(self) -> dict[str, Any]:
        gr = self.gamma_range
        gamma: Any = gr[0] if gr[0] == gr[1] else list(gr)
        return {
            "name": self.__class__.__name__,
            "params": {
                "gamma": gamma,
                "invert": self.invert,
                "p": self.p,
            },
        }
