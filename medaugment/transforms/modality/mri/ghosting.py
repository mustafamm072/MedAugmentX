"""MRI phase-encoding ghosting artifact."""
from __future__ import annotations

from typing import Any, Union

import numpy as np

from medaugment.core.base import Transform
from medaugment.core.utils import SeedLike, as_float32
from medaugment.core.volume import MedVolume

Range = Union[float, tuple[float, float]]
IntRange = Union[int, tuple[int, int]]


class GhostingArtifact(Transform):
    """Simulate MRI phase-encoding ghosting.

    Ghosting arises from periodic motion during the phase-encode direction
    acquisition — cardiac pulsation, breathing, or patient movement produce
    attenuated shifted replicas of the anatomy overlaid on the image.

    Implementation: add one or more rolled, attenuated copies of the image
    along the phase-encode axis.  The ghost intensity is expressed relative
    to the image maximum so the effect is consistent across intensity scales.

    Works on 2-D and 3-D inputs.  For 3-D, ghosting is applied in each axial
    plane (i.e. along the in-plane axis selected by ``phase_encode_axis``).

    Args:
        ghost_intensity: Ghost-to-original amplitude ratio.  Scalar → fixed;
            tuple → sampled from ``(lo, hi)``.  Typical MRI values: 0.05–0.20.
        ghost_shift: Shift in pixels.  Scalar → fixed; tuple → sampled.
        phase_encode_axis: ``"y"`` (row, default) or ``"x"`` (column).
        num_ghosts: Number of independent ghost replicas to superimpose.
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        ghost_intensity: Range = (0.05, 0.15),
        ghost_shift: IntRange = (8, 32),
        phase_encode_axis: str = "y",
        num_ghosts: int = 1,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if isinstance(ghost_intensity, (int, float)):
            v = float(ghost_intensity)
            if v < 0:
                raise ValueError("ghost_intensity must be >= 0")
            self.ghost_intensity_range: tuple[float, float] = (v, v)
        else:
            lo, hi = float(ghost_intensity[0]), float(ghost_intensity[1])
            if lo < 0 or hi < lo:
                raise ValueError(f"ghost_intensity range invalid: {ghost_intensity}")
            self.ghost_intensity_range = (lo, hi)

        if isinstance(ghost_shift, int):
            self.ghost_shift_range: tuple[int, int] = (int(ghost_shift), int(ghost_shift))
        else:
            lo_i, hi_i = int(ghost_shift[0]), int(ghost_shift[1])
            if lo_i < 0 or hi_i < lo_i:
                raise ValueError(f"ghost_shift range invalid: {ghost_shift}")
            self.ghost_shift_range = (lo_i, hi_i)

        if phase_encode_axis not in ("x", "y"):
            raise ValueError("phase_encode_axis must be 'x' or 'y'")
        self.phase_encode_axis = phase_encode_axis

        if num_ghosts < 1:
            raise ValueError("num_ghosts must be >= 1")
        self.num_ghosts = int(num_ghosts)

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)
        # "y" → second-to-last axis; "x" → last axis
        axis = -2 if self.phase_encode_axis == "y" else -1

        img_max = float(np.abs(image).max())
        if img_max < 1e-8:
            return volume

        out = image.copy()
        for _ in range(self.num_ghosts):
            intensity = float(self.rng.uniform(*self.ghost_intensity_range))
            shift = int(
                self.rng.integers(self.ghost_shift_range[0], self.ghost_shift_range[1] + 1)
            )
            if shift == 0:
                continue
            direction = 1 if self.rng.random() < 0.5 else -1
            ghost = np.roll(image, direction * shift, axis=axis).astype(np.float32)
            out = out + ghost * intensity

        return volume.replace(image=out)

    def to_dict(self) -> dict[str, Any]:
        gi = self.ghost_intensity_range
        gs = self.ghost_shift_range
        return {
            "name": self.__class__.__name__,
            "params": {
                "ghost_intensity": gi[0] if gi[0] == gi[1] else list(gi),
                "ghost_shift": gs[0] if gs[0] == gs[1] else list(gs),
                "phase_encode_axis": self.phase_encode_axis,
                "num_ghosts": self.num_ghosts,
                "p": self.p,
            },
        }
