"""SlabShift — Z-axis recon-centre variation for DBT volumes."""
from __future__ import annotations

from typing import Union

import numpy as np

from medaugment.core.base import Transform
from medaugment.core.utils import SeedLike
from medaugment.core.volume import MedVolume

IntRange = Union[int, tuple[int, int]]


class SlabShift(Transform):
    """Shift the entire volume along the slice (Z) axis.

    Simulates inter-scan variation in the reconstruction centre between two
    DBT acquisitions of the same patient. The number of slices and the
    voxel spacing are preserved; gaps introduced by the shift are filled
    with ``cval`` (default zero, i.e. air).

    Args:
        max_shift: Maximum absolute shift in slices. Either an int (treated
            as ``(-max_shift, max_shift)``) or a ``(low, high)`` tuple.
        cval: Fill value for newly empty slices. Defaults to ``0.0``.
        p: Probability of applying.
        seed: RNG seed.

    Raises:
        ValueError: If applied to a 2D volume — DBT is 3D by definition.
    """

    def __init__(
        self,
        max_shift: IntRange = 2,
        cval: float = 0.0,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if isinstance(max_shift, int):
            self.shift_range: tuple[int, int] = (-int(max_shift), int(max_shift))
        else:
            self.shift_range = (int(max_shift[0]), int(max_shift[1]))
        if self.shift_range[0] > self.shift_range[1]:
            raise ValueError(f"shift range invalid: {self.shift_range}")
        self.cval = float(cval)

    def _shift_array(self, arr: np.ndarray, shift: int, cval) -> np.ndarray:
        out = np.full_like(arr, fill_value=cval)
        depth = arr.shape[0]
        if shift == 0:
            return arr.copy()
        if shift > 0:
            n = max(0, depth - shift)
            if n > 0:
                out[shift:shift + n] = arr[:n]
        else:
            n = max(0, depth + shift)  # shift is negative
            if n > 0:
                out[:n] = arr[-shift:-shift + n]
        return out

    def apply(self, volume: MedVolume) -> MedVolume:
        if not volume.is_3d:
            raise ValueError("SlabShift requires a 3D volume")
        shift = int(self.rng.integers(self.shift_range[0], self.shift_range[1] + 1))
        if shift == 0:
            return volume
        new_image = self._shift_array(volume.image, shift, self.cval)
        new_mask = (
            None
            if volume.mask is None
            else self._shift_array(volume.mask, shift, 0).astype(volume.mask.dtype, copy=False)
        )
        return volume.replace(image=new_image, mask=new_mask)
