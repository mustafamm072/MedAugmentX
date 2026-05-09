"""Additive brightness shift and multiplicative contrast adjustment."""
from __future__ import annotations

from typing import Any, Union

import numpy as np

from medaugment.core.base import Transform
from medaugment.core.utils import SeedLike, as_float32
from medaugment.core.volume import MedVolume

Range = Union[float, tuple[float, float]]


def _as_symmetric_range(x: Range, name: str) -> tuple[float, float]:
    """Scalar → ``(-v, +v)``; tuple → validated ``(lo, hi)``."""
    if isinstance(x, (int, float)):
        v = float(x)
        return (-v, v)
    lo, hi = float(x[0]), float(x[1])
    if lo > hi:
        raise ValueError(f"{name}: lower bound {lo} > upper bound {hi}")
    return lo, hi


class BrightnessContrast(Transform):
    """Additive brightness shift + multiplicative contrast factor.

    Applies:  ``out = clip(image * contrast + brightness, *clip)``

    Both parameters operate in the image's **native intensity space** — no
    normalisation is applied.  Scale the ranges to your dataset:

    * Normalised ``[0, 1]`` images: ``brightness=0.05``, ``contrast=(0.9, 1.1)``
    * CT in Hounsfield units: ``brightness=30.0``, ``contrast=(0.95, 1.05)``

    The mask is never modified.

    Args:
        brightness: Additive offset range.  Scalar → symmetric ``(-v, +v)``;
            tuple → explicit ``(lo, hi)``.
        contrast: Multiplicative factor range.  Scalar → ``(1-v, 1+v)``
            centred on identity; tuple → explicit ``(lo, hi)``.  Must be > 0.
        clip: Optional ``(min, max)`` to clamp the result.
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        brightness: Range = 0.0,
        contrast: Range = (0.9, 1.1),
        clip: tuple[float, float] | None = None,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.brightness_range = _as_symmetric_range(brightness, "brightness")

        if isinstance(contrast, (int, float)):
            v = float(contrast)
            if v <= 0:
                raise ValueError(f"contrast scalar must be > 0, got {v}")
            self.contrast_range: tuple[float, float] = (v, v)
        else:
            lo, hi = float(contrast[0]), float(contrast[1])
            if lo <= 0 or hi < lo:
                raise ValueError(f"contrast range must be positive and ordered: {contrast}")
            self.contrast_range = (lo, hi)

        self.clip = clip

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)
        brightness = float(self.rng.uniform(*self.brightness_range))
        contrast = float(self.rng.uniform(*self.contrast_range))
        out = image * contrast + brightness
        if self.clip is not None:
            out = np.clip(out, self.clip[0], self.clip[1])
        return volume.replace(image=out.astype(np.float32, copy=False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {
                "brightness": list(self.brightness_range),
                "contrast": list(self.contrast_range),
                "clip": list(self.clip) if self.clip is not None else None,
                "p": self.p,
            },
        }
