"""CT metal streak artifact simulation."""
from __future__ import annotations

from typing import Any, Union

import numpy as np

from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike, as_float32
from medaugmentx.core.volume import MedVolume

Range = Union[float, tuple[float, float]]
IntRange = Union[int, tuple[int, int]]


class MetalStreak(Transform):
    """Simulate CT metal streak artifacts.

    Dense implants (dental fillings, hip prostheses, surgical clips) attenuate
    the X-ray beam so strongly that the reconstruction produces bright and dark
    streaks radiating from the metal.  This transform adds a star-burst pattern
    of streaks emanating from one or more seed points, with a radial fall-off so
    the effect is strongest near the metal.

    Applied per axial slice; the same seed point and streak pattern are used on
    every slice of a 3-D volume.  The mask is never modified.

    Args:
        intensity: Streak amplitude relative to the image intensity range.
            Scalar → fixed; tuple ``(lo, hi)`` → sampled per call.  Typical
            values ``0.1–0.4``.
        num_streaks: Number of angular streak lobes in the star-burst.
            Int or ``(low, high)`` inclusive range.
        num_sources: Number of independent metal seed points.
        falloff: Radial decay scale as a fraction of the image half-diagonal
            (smaller → streaks stay close to the metal).
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        intensity: Range = (0.1, 0.3),
        num_streaks: IntRange = (6, 12),
        num_sources: IntRange = 1,
        falloff: float = 0.5,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if isinstance(intensity, (int, float)):
            v = float(intensity)
            if v < 0:
                raise ValueError("intensity must be >= 0")
            self.intensity_range: tuple[float, float] = (v, v)
        else:
            lo, hi = float(intensity[0]), float(intensity[1])
            if lo < 0 or hi < lo:
                raise ValueError(f"intensity range invalid: {intensity}")
            self.intensity_range = (lo, hi)
        self.streak_range = _as_int_range(num_streaks, "num_streaks", min_value=1)
        self.source_range = _as_int_range(num_sources, "num_sources", min_value=1)
        if falloff <= 0:
            raise ValueError("falloff must be > 0")
        self.falloff = float(falloff)

    def _streak_field(self, h: int, w: int) -> np.ndarray:
        yy, xx = np.meshgrid(
            np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
        )
        diag = float(np.hypot(h, w))
        n_sources = int(self.rng.integers(self.source_range[0], self.source_range[1] + 1))
        field = np.zeros((h, w), dtype=np.float32)
        for _ in range(n_sources):
            cy = float(self.rng.uniform(0.25 * h, 0.75 * h))
            cx = float(self.rng.uniform(0.25 * w, 0.75 * w))
            n_streaks = int(self.rng.integers(self.streak_range[0], self.streak_range[1] + 1))
            phase = float(self.rng.uniform(0, 2 * np.pi))
            theta = np.arctan2(yy - cy, xx - cx)
            r = np.hypot(yy - cy, xx - cx)
            angular = np.cos(n_streaks * theta + phase)
            radial = np.exp(-r / (self.falloff * diag + 1e-8))
            field += angular * radial
        return field

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)
        img_range = float(image.max()) - float(image.min())
        if img_range < 1e-8:
            return volume
        intensity = float(self.rng.uniform(*self.intensity_range))
        if intensity == 0.0:
            return volume

        h, w = image.shape[-2], image.shape[-1]
        field = self._streak_field(h, w) * intensity * img_range
        if volume.is_3d:
            out = image + field[np.newaxis, :, :]
        else:
            out = image + field
        return volume.replace(image=out.astype(np.float32, copy=False))

    def to_dict(self) -> dict[str, Any]:
        ir, sr, so = self.intensity_range, self.streak_range, self.source_range
        return {
            "name": self.__class__.__name__,
            "params": {
                "intensity": ir[0] if ir[0] == ir[1] else list(ir),
                "num_streaks": sr[0] if sr[0] == sr[1] else list(sr),
                "num_sources": so[0] if so[0] == so[1] else list(so),
                "falloff": self.falloff,
                "p": self.p,
            },
        }


def _as_int_range(x: IntRange, name: str, min_value: int) -> tuple[int, int]:
    if isinstance(x, int):
        if x < min_value:
            raise ValueError(f"{name} must be >= {min_value}, got {x}")
        return (int(x), int(x))
    lo, hi = int(x[0]), int(x[1])
    if lo < min_value or hi < lo:
        raise ValueError(f"{name} range invalid: {x}")
    return lo, hi
