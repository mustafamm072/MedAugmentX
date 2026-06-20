"""Anti-scatter grid line artifact simulation."""
from __future__ import annotations

from typing import Any, Union

import numpy as np

from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike, as_float32
from medaugmentx.core.volume import MedVolume

Range = Union[float, tuple[float, float]]


class GridArtifact(Transform):
    """Simulate stationary anti-scatter grid lines.

    A non-reciprocating (stationary) anti-scatter grid leaves fine periodic
    line shadows across the radiograph.  This transform multiplies the image
    by a sinusoidal stripe pattern oriented along one in-plane axis.

    ``out = image * (1 + amplitude * sin(2*pi*frequency*coord + phase))``

    The mask is never modified.

    Args:
        amplitude: Peak relative modulation of the grid lines.  Scalar →
            fixed; tuple ``(lo, hi)`` → sampled per call.  Typical values
            ``0.02–0.12``.
        frequency: Grid spatial frequency in cycles per pixel.  Scalar →
            fixed; tuple → sampled.  ``0.25`` ≈ a line every 4 pixels.
        axis: Stripe orientation — ``"x"`` for vertical lines (varying along
            columns) or ``"y"`` for horizontal lines (varying along rows).
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        amplitude: Range = (0.03, 0.1),
        frequency: Range = (0.2, 0.45),
        axis: str = "x",
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.amplitude_range = _as_range(amplitude, "amplitude", min_value=0.0)
        self.frequency_range = _as_range(frequency, "frequency", min_value=0.0)
        if self.frequency_range[0] <= 0:
            raise ValueError("frequency must be > 0")
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        self.axis = axis

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)
        amplitude = float(self.rng.uniform(*self.amplitude_range))
        frequency = float(self.rng.uniform(*self.frequency_range))
        phase = float(self.rng.uniform(0, 2 * np.pi))
        if amplitude == 0.0:
            return volume

        # "x" → varies along the last axis (columns); "y" → second-to-last (rows).
        axis = image.ndim - 1 if self.axis == "x" else image.ndim - 2
        n = image.shape[axis]
        coord = np.arange(n, dtype=np.float32)
        wave = 1.0 + amplitude * np.sin(2 * np.pi * frequency * coord + phase)
        shape = [1] * image.ndim
        shape[axis] = n
        out = image * wave.reshape(shape)
        return volume.replace(image=out.astype(np.float32, copy=False))

    def to_dict(self) -> dict[str, Any]:
        ar, fr = self.amplitude_range, self.frequency_range
        return {
            "name": self.__class__.__name__,
            "params": {
                "amplitude": ar[0] if ar[0] == ar[1] else list(ar),
                "frequency": fr[0] if fr[0] == fr[1] else list(fr),
                "axis": self.axis,
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
