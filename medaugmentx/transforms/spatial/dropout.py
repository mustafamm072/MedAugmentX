"""CoarseDropout — random rectangular/box occlusion (cutout)."""
from __future__ import annotations

from typing import Any, Union

from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike
from medaugmentx.core.volume import MedVolume

IntRange = Union[int, tuple[int, int]]


def _as_int_range(x: IntRange, name: str) -> tuple[int, int]:
    if isinstance(x, int):
        if x < 0:
            raise ValueError(f"{name} must be >= 0, got {x}")
        return (int(x), int(x))
    lo, hi = int(x[0]), int(x[1])
    if lo < 0 or hi < lo:
        raise ValueError(f"{name} range invalid: {x}")
    return lo, hi


class CoarseDropout(Transform):
    """Zero out a random number of rectangular (2D) or box (3D) regions.

    Also known as *cutout*.  Forces the network to use the whole field of
    view rather than over-relying on a single salient region — a strong
    regulariser for both classification and segmentation.  Hole sizes are
    expressed as a fraction of each axis so the transform is
    resolution-independent and works on 2-D and 3-D volumes alike.

    The mask is left untouched by default (the ground-truth annotation is
    still valid behind an occlusion); pass ``fill_mask=True`` to also blank
    the mask inside each hole.

    Args:
        num_holes: Number of holes to drop.  Int (fixed) or ``(low, high)``
            inclusive range sampled per call.
        hole_size: Fractional extent of each hole along every axis, as
            ``(min_frac, max_frac)`` in ``(0, 1]``.  Each hole samples an
            independent size per axis.
        fill_value: Value written inside the holes (default ``0.0``).
        fill_mask: If True, also set the mask to ``0`` inside each hole.
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        num_holes: IntRange = (1, 4),
        hole_size: tuple[float, float] = (0.05, 0.2),
        fill_value: float = 0.0,
        fill_mask: bool = False,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.num_range = _as_int_range(num_holes, "num_holes")
        lo, hi = float(hole_size[0]), float(hole_size[1])
        if not (0.0 < lo <= hi <= 1.0):
            raise ValueError(f"hole_size must satisfy 0 < lo <= hi <= 1, got {hole_size}")
        self.hole_size: tuple[float, float] = (lo, hi)
        self.fill_value = float(fill_value)
        self.fill_mask = bool(fill_mask)

    def apply(self, volume: MedVolume) -> MedVolume:
        shape = volume.shape
        n = int(self.rng.integers(self.num_range[0], self.num_range[1] + 1))
        if n == 0:
            return volume

        new_image = volume.image.copy()
        new_mask = None
        if volume.mask is not None and self.fill_mask:
            new_mask = volume.mask.copy()

        for _ in range(n):
            slices = []
            for dim in shape:
                frac = float(self.rng.uniform(*self.hole_size))
                size = max(1, int(round(frac * dim)))
                size = min(size, dim)
                origin = int(self.rng.integers(0, dim - size + 1))
                slices.append(slice(origin, origin + size))
            region = tuple(slices)
            new_image[region] = self.fill_value
            if new_mask is not None:
                new_mask[region] = 0

        return volume.replace(image=new_image, mask=new_mask)

    def to_dict(self) -> dict[str, Any]:
        nr = self.num_range
        return {
            "name": self.__class__.__name__,
            "params": {
                "num_holes": nr[0] if nr[0] == nr[1] else list(nr),
                "hole_size": list(self.hole_size),
                "fill_value": self.fill_value,
                "fill_mask": self.fill_mask,
                "p": self.p,
            },
        }
