"""ReconStreak — DBT out-of-plane reconstruction streak artifact."""
from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy.ndimage import shift as nd_shift

from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike, as_float32
from medaugmentx.core.volume import MedVolume

Range = Union[float, tuple[float, float]]
IntRange = Union[int, tuple[int, int]]


class ReconStreak(Transform):
    """Simulate DBT out-of-plane reconstruction streaks.

    Because tomosynthesis reconstructs from a limited scan arc, high-contrast
    structures are not fully localised to their true slice: they reappear as
    faded, in-plane-shifted replicas in neighbouring slices (the parallax of
    limited-angle reconstruction).  This transform superimposes such shifted,
    attenuated copies from adjacent planes onto each slice.

    The mask is treated as an intensity-artifact source and is never modified
    — the true anatomy annotation stays valid.

    Args:
        amplitude: Strength of the spilled replicas relative to the source.
            Scalar → fixed; tuple ``(lo, hi)`` → sampled per call.  Typical
            values ``0.05–0.25``.
        num_planes: Number of neighbouring planes (on each side) to spill
            from.  Int or ``(low, high)`` inclusive range.
        displacement: In-plane shift (pixels) per plane of separation — the
            parallax slope.
        decay: Per-plane geometric fall-off of the replica amplitude.
        axis: In-plane direction of the parallax shift — ``"x"`` (default) or
            ``"y"``.
        p: Probability of applying.
        seed: RNG seed.

    Raises:
        ValueError: If applied to a 2D volume — DBT is 3D by definition.
    """

    def __init__(
        self,
        amplitude: Range = (0.05, 0.2),
        num_planes: IntRange = (1, 3),
        displacement: float = 1.5,
        decay: float = 0.6,
        axis: str = "x",
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if isinstance(amplitude, (int, float)):
            v = float(amplitude)
            if v < 0:
                raise ValueError("amplitude must be >= 0")
            self.amplitude_range: tuple[float, float] = (v, v)
        else:
            lo, hi = float(amplitude[0]), float(amplitude[1])
            if lo < 0 or hi < lo:
                raise ValueError(f"amplitude range invalid: {amplitude}")
            self.amplitude_range = (lo, hi)

        if isinstance(num_planes, int):
            plo = phi = int(num_planes)
        else:
            plo, phi = int(num_planes[0]), int(num_planes[1])
        if plo < 1 or phi < plo:
            raise ValueError(f"num_planes range invalid: {num_planes}")
        self.plane_range: tuple[int, int] = (plo, phi)

        self.displacement = float(displacement)
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.decay = float(decay)
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        self.axis = axis

    def apply(self, volume: MedVolume) -> MedVolume:
        if not volume.is_3d:
            raise ValueError("ReconStreak requires a 3D volume")
        image = as_float32(volume.image)
        amplitude = float(self.rng.uniform(*self.amplitude_range))
        if amplitude == 0.0:
            return volume
        n = int(self.rng.integers(self.plane_range[0], self.plane_range[1] + 1))

        # In-plane axis index: "x" → last, "y" → second-to-last.
        ip_axis = 2 if self.axis == "x" else 1
        out = image.copy()
        for k in range(1, n + 1):
            weight = amplitude * (self.decay ** (k - 1))
            disp = k * self.displacement
            for direction in (+1, -1):
                vec = [0.0, 0.0, 0.0]
                vec[0] = float(direction * k)  # plane (Z) offset
                vec[ip_axis] = float(direction * disp)  # in-plane parallax shift
                replica = nd_shift(image, vec, order=1, mode="constant", cval=0.0)
                out += weight * replica
        return volume.replace(image=out.astype(np.float32, copy=False))

    def to_dict(self) -> dict[str, Any]:
        ar, pr = self.amplitude_range, self.plane_range
        return {
            "name": self.__class__.__name__,
            "params": {
                "amplitude": ar[0] if ar[0] == ar[1] else list(ar),
                "num_planes": pr[0] if pr[0] == pr[1] else list(pr),
                "displacement": self.displacement,
                "decay": self.decay,
                "axis": self.axis,
                "p": self.p,
            },
        }
