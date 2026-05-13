"""Anatomy-aware random cropping."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike
from medaugmentx.core.volume import MedVolume


class AnatomicCrop(Transform):
    """Random crop biased toward foreground (non-zero) anatomy.

    Behaviour:

    1. With probability ``foreground_prob`` (and only when a non-empty mask
       or foreground exists) the crop is sampled to ensure at least one
       foreground voxel lies inside the patch.
    2. Otherwise a uniform random crop is taken.

    If the requested ``size`` is larger than the volume on any axis the
    volume is returned unchanged on that axis (centre-aligned, no padding).

    Args:
        size: Target crop size per axis. Must match the volume ndim.
        foreground_prob: Probability of biasing the sample toward foreground.
        foreground_threshold: Pixel intensity above which a voxel counts as
            foreground when no mask is present.
        p: Probability of applying the transform.
        seed: RNG seed.
    """

    def __init__(
        self,
        size: Sequence[int],
        foreground_prob: float = 0.5,
        foreground_threshold: float = 0.0,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.size: tuple[int, ...] = tuple(int(s) for s in size)
        if not 0.0 <= float(foreground_prob) <= 1.0:
            raise ValueError(f"foreground_prob must be in [0, 1], got {foreground_prob}")
        self.foreground_prob = float(foreground_prob)
        self.foreground_threshold = float(foreground_threshold)

    def _sample_origin(
        self,
        shape: tuple[int, ...],
        size: tuple[int, ...],
        foreground_mask: np.ndarray | None,
    ) -> tuple[int, ...]:
        bias = (
            foreground_mask is not None
            and foreground_mask.any()
            and self.rng.random() < self.foreground_prob
        )
        if bias:
            assert foreground_mask is not None
            fg_indices = np.argwhere(foreground_mask)
            anchor = fg_indices[int(self.rng.integers(0, len(fg_indices)))]
            origin = []
            for ax, anc in enumerate(anchor):
                lo = max(0, int(anc) - size[ax] + 1)
                hi = min(shape[ax] - size[ax], int(anc))
                if lo > hi:  # crop bigger than axis
                    origin.append(0)
                else:
                    origin.append(int(self.rng.integers(lo, hi + 1)))
            return tuple(origin)

        origin = []
        for ax, sz in enumerate(size):
            limit = shape[ax] - sz
            if limit <= 0:
                origin.append(0)
            else:
                origin.append(int(self.rng.integers(0, limit + 1)))
        return tuple(origin)

    def apply(self, volume: MedVolume) -> MedVolume:
        if len(self.size) != volume.ndim:
            raise ValueError(
                f"crop size ndim ({len(self.size)}) does not match volume ndim ({volume.ndim})"
            )
        size = tuple(min(s, sh) for s, sh in zip(self.size, volume.shape))

        if volume.mask is not None:
            fg = volume.mask > 0
        else:
            fg = volume.image > self.foreground_threshold

        origin = self._sample_origin(volume.shape, size, fg)
        slices = tuple(slice(o, o + s) for o, s in zip(origin, size))
        new_image = volume.image[slices].copy()
        new_mask = None if volume.mask is None else volume.mask[slices].copy()
        return volume.replace(image=new_image, mask=new_mask)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {
                "size": list(self.size),
                "foreground_prob": self.foreground_prob,
                "foreground_threshold": self.foreground_threshold,
                "p": self.p,
            },
        }
