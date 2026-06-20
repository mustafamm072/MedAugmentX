"""MRI in-plane rigid motion artifact."""
from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy.ndimage import rotate, shift

from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike, as_float32
from medaugmentx.core.volume import MedVolume

Range = Union[float, tuple[float, float]]
IntRange = Union[int, tuple[int, int]]


class MRIMotion(Transform):
    """Simulate MRI in-plane rigid-body motion (blur + ghosting).

    During a scan the patient occupies several slightly different positions.
    Because k-space is filled across the whole acquisition, the reconstructed
    image is effectively a weighted average of those positions, producing the
    characteristic motion blur and edge ghosting seen on clinical MRI.

    This transform averages the original image with a few mildly
    rotated/translated copies (in-plane only).  For 3-D volumes the same
    in-plane motion is applied to every slice.  The mask is treated as an
    intensity artifact source and is **not** moved — the true anatomy
    annotation stays valid.

    Args:
        degrees: In-plane rotation magnitude per motion state, in degrees.
            Scalar → symmetric ``(-v, +v)``; tuple → explicit ``(lo, hi)``.
        translation: In-plane translation per motion state, in pixels.
            Scalar → symmetric; tuple → explicit ``(lo, hi)``.
        num_movements: Number of additional motion states to blend in.
            Int (fixed) or ``(low, high)`` inclusive range.
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        degrees: Range = (1.0, 5.0),
        translation: Range = (1.0, 4.0),
        num_movements: IntRange = (1, 3),
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.degrees_range = _signed_range(degrees, "degrees")
        self.translation_range = _signed_range(translation, "translation")
        if isinstance(num_movements, int):
            lo = hi = int(num_movements)
        else:
            lo, hi = int(num_movements[0]), int(num_movements[1])
        if lo < 1 or hi < lo:
            raise ValueError(f"num_movements range invalid: {num_movements}")
        self.num_range: tuple[int, int] = (lo, hi)

    def _move(self, image: np.ndarray, angle: float, dy: float, dx: float) -> np.ndarray:
        # In-plane axes are the last two for both 2-D and 3-D arrays.
        axes = (image.ndim - 2, image.ndim - 1)
        rotated = rotate(
            image, angle, axes=axes, reshape=False, order=1, mode="reflect"
        )
        sh = [0.0] * image.ndim
        sh[-2] = dy
        sh[-1] = dx
        return shift(rotated, sh, order=1, mode="reflect")

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)
        n = int(self.rng.integers(self.num_range[0], self.num_range[1] + 1))

        acc = image.copy()
        for _ in range(n):
            angle = float(self.rng.uniform(*self.degrees_range))
            dy = float(self.rng.uniform(*self.translation_range))
            dx = float(self.rng.uniform(*self.translation_range))
            acc += self._move(image, angle, dy, dx)
        out = acc / float(n + 1)
        return volume.replace(image=out.astype(np.float32, copy=False))

    def to_dict(self) -> dict[str, Any]:
        dr, tr, nr = self.degrees_range, self.translation_range, self.num_range
        return {
            "name": self.__class__.__name__,
            "params": {
                "degrees": (-dr[0]) if dr[0] == -dr[1] else list(dr),
                "translation": (-tr[0]) if tr[0] == -tr[1] else list(tr),
                "num_movements": nr[0] if nr[0] == nr[1] else list(nr),
                "p": self.p,
            },
        }


def _signed_range(x: Range, name: str) -> tuple[float, float]:
    if isinstance(x, (int, float)):
        v = float(x)
        if v < 0:
            raise ValueError(f"{name} scalar must be >= 0, got {v}")
        return (-v, v)
    lo, hi = float(x[0]), float(x[1])
    if lo > hi:
        raise ValueError(f"{name} range invalid: {x}")
    return lo, hi
