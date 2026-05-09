"""SliceDropout — randomly zero a small number of slices."""
from __future__ import annotations

from typing import Any, Union

from medaugment.core.base import Transform
from medaugment.core.utils import SeedLike
from medaugment.core.volume import MedVolume

IntRange = Union[int, tuple[int, int]]


class SliceDropout(Transform):
    """Zero one or more slices along the Z axis.

    This is a robustness augmentation: models trained with occasional
    blanked slices learn not to assume strict per-slice continuity, which
    matters when DBT reconstructions occasionally drop frames or contain
    severe streak artifacts on isolated slices.

    Args:
        num_slices: How many slices to drop. Either an int (fixed count)
            or a ``(low, high)`` inclusive range to sample from.
        cval: Fill value (defaults to ``0.0``).
        affect_mask: If True, also zero the mask on the dropped slices.
            Defaults to False — most segmentation pipelines want the
            ground truth preserved.
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        num_slices: IntRange = 1,
        cval: float = 0.0,
        affect_mask: bool = False,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if isinstance(num_slices, int):
            self.num_range: tuple[int, int] = (int(num_slices), int(num_slices))
        else:
            self.num_range = (int(num_slices[0]), int(num_slices[1]))
        if self.num_range[0] < 0 or self.num_range[1] < self.num_range[0]:
            raise ValueError(f"num_slices range invalid: {num_slices}")
        self.cval = float(cval)
        self.affect_mask = bool(affect_mask)

    def apply(self, volume: MedVolume) -> MedVolume:
        if not volume.is_3d:
            raise ValueError("SliceDropout requires a 3D volume")
        depth = volume.shape[0]
        if depth == 0:
            return volume
        n = int(self.rng.integers(self.num_range[0], self.num_range[1] + 1))
        n = min(n, depth)
        if n == 0:
            return volume

        idxs = self.rng.choice(depth, size=n, replace=False)
        new_image = volume.image.copy()
        new_image[idxs] = self.cval

        new_mask = None
        if volume.mask is not None and self.affect_mask:
            new_mask = volume.mask.copy()
            new_mask[idxs] = 0
        return volume.replace(image=new_image, mask=new_mask)

    def to_dict(self) -> dict[str, Any]:
        nr = self.num_range
        num_slices: Any = nr[0] if nr[0] == nr[1] else list(nr)
        return {
            "name": self.__class__.__name__,
            "params": {
                "num_slices": num_slices,
                "cval": self.cval,
                "affect_mask": self.affect_mask,
                "p": self.p,
            },
        }
