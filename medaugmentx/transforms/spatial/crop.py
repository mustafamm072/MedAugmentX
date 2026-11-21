"""Anatomy-aware random cropping."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np

from medaugmentx.core import geometry
from medaugmentx.core.base import Transform
from medaugmentx.core.parameters import positive_size, validate_parameters
from medaugmentx.core.physical import grid_translation
from medaugmentx.core.utils import SeedLike, as_float32
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
        validate_parameters(
            size=size, foreground_prob=foreground_prob, foreground_threshold=foreground_threshold
        )
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
        anchor = None
        if bias:
            assert foreground_mask is not None
            indices = np.flatnonzero(foreground_mask)
            anchor = np.unravel_index(indices[int(self.rng.integers(len(indices)))], shape)
        return _sample_crop_origin(self.rng, shape, size, anchor)

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
        return _crop(volume, origin, size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {
                "size": list(self.size),
                "foreground_prob": self.foreground_prob,
                "foreground_threshold": self.foreground_threshold,
                "p": self.p,
                "seed": self._seed,
            },
        }


class ClassBalancedCrop(Transform):
    """Extract a fixed-size training patch, optionally anchored on a mask class.

    The foreground branch picks uniformly among the eligible classes present
    in this volume, then uniformly among voxels of that class. The crop origin
    is sampled uniformly from origins that include the selected voxel. If this
    branch is skipped, the mask is missing or no eligible class is present,
    origins are sampled uniformly across the image. This branch can also include
    foreground. Class balancing applies to the anchor within each image.

    Short axes are padded symmetrically, with an odd extra voxel at the high
    end. Mask padding uses label 0. Images are converted to float32 when padding
    is needed. Spacing is preserved, and the affine and targets follow the patch
    origin. Targets outside the patch keep their translated coordinates.

    Args:
        size: Fixed patch shape of positive voxel counts in array-axis order.
        foreground_prob: Probability of taking the class-anchored branch.
        labels: Eligible nonzero integer label IDs, or None for all nonzero
            classes. Supply IDs explicitly to exclude ignore labels.
        cval: Image padding value in the current intensity units.
        p: Probability of applying. Use 1 for guaranteed patch shape.
        seed: RNG seed.
    """

    def __init__(
        self,
        size: Sequence[int],
        foreground_prob: float = 0.5,
        *,
        labels: Sequence[int] | None = None,
        cval: float = 0.0,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.size = positive_size(size)
        if len(self.size) not in (2, 3):
            raise ValueError("size must have two or three entries")
        self.foreground_prob = float(foreground_prob)
        if not 0 <= self.foreground_prob <= 1:
            raise ValueError("foreground_prob must be in [0, 1]")
        self.cval = float(cval)
        if not np.isfinite(self.cval) or abs(self.cval) > float(np.finfo(np.float32).max):
            raise ValueError("cval must be finite and representable as float32")
        self.labels = None
        if labels is not None:
            values = tuple(labels)
            if not values or any(
                isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v == 0
                for v in values
            ):
                raise ValueError("labels must contain nonzero integers")
            self.labels = tuple(sorted(set(int(v) for v in values)))

    def apply(self, volume: MedVolume) -> MedVolume:
        if len(self.size) != volume.ndim:
            raise ValueError("patch size ndim must match volume ndim")
        source_size = tuple(min(s, n) for s, n in zip(self.size, volume.shape))
        anchor = None
        if volume.mask is not None and self.rng.random() < self.foreground_prob:
            # Compare IDs as Python integers to avoid common-type coercion of
            # signed, unsigned and integer-valued floating masks.
            eligible = None if self.labels is None else set(self.labels)
            present = [
                label
                for label in np.unique(volume.mask)
                if label != 0 and (eligible is None or int(label) in eligible)
            ]
            if present:
                label = present[int(self.rng.integers(len(present)))]
                indices = np.flatnonzero(volume.mask == label)
                flat = indices[int(self.rng.integers(len(indices)))]
                anchor = np.unravel_index(flat, volume.shape)

        start = _sample_crop_origin(self.rng, volume.shape, source_size, anchor)
        origin = tuple(o - (s - n) // 2 for o, s, n in zip(start, self.size, source_size))
        return _crop(volume, origin, self.size, cval=self.cval)


def _sample_crop_origin(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    size: tuple[int, ...],
    anchor: tuple[int, ...] | None,
) -> tuple[int, ...]:
    origin = []
    for ax, (length, width) in enumerate(zip(shape, size)):
        limit = length - width
        if anchor is not None:
            lo = max(0, int(anchor[ax]) - width + 1)
            hi = min(limit, int(anchor[ax]))
            origin.append(int(rng.integers(lo, hi + 1)))
        else:
            origin.append(int(rng.integers(limit + 1)) if limit > 0 else 0)
    return tuple(origin)


def _crop(
    volume: MedVolume, origin: tuple[int, ...], size: tuple[int, ...], *, cval: float = 0.0
) -> MedVolume:
    ends = tuple(start + width for start, width in zip(origin, size))
    region = tuple(
        slice(max(0, start), min(end, length))
        for start, end, length in zip(origin, ends, volume.shape)
    )
    mask = None if volume.mask is None else volume.mask[region]
    widths = [
        (max(0, -start), max(0, end - length))
        for start, end, length in zip(origin, ends, volume.shape)
    ]
    padded = any(before or after for before, after in widths)

    def crop_image(array: np.ndarray) -> np.ndarray:
        cropped = array[region]
        if padded:
            return np.pad(as_float32(cropped), widths, constant_values=cval)
        return cropped.copy()

    if mask is not None:
        mask = np.pad(mask, widths, constant_values=0) if padded else mask.copy()
    return volume.warp(
        geometry.translate_map(-np.asarray(origin, dtype=np.float64)),
        image_map=crop_image,
        mask=mask,
        grid_map=grid_translation(origin),
    )
