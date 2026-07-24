"""Deterministic geometry utilities: Resize, Pad, CenterCrop.

These are not random augmentations — they are shape-normalisation helpers
that pair naturally with the random transforms in a pipeline (e.g. resize to
a common training size, or pad/crop to a fixed patch shape for batching).
All three keep the mask in lockstep with the image: resampling uses
nearest-neighbour for the mask, padding fills the mask with ``0``, and
cropping slices image and mask identically.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.ndimage import zoom

from medaugmentx.core import geometry
from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike, as_float32
from medaugmentx.core.volume import MedVolume


class Resize(Transform):
    """Resample the volume to a fixed ``size`` (one entry per spatial axis).

    The image is resampled with spline interpolation of order ``order``; the
    mask always uses nearest-neighbour (``order=0``) so label values are
    preserved.  ``spacing`` is rescaled to reflect the new voxel grid.

    Args:
        size: Target shape, matching the volume ndim (2D or 3D).
        order: Spline order for the image (0–3).  Mask is always ``order=0``.
        p: Probability of applying.
        seed: RNG seed (unused — Resize is deterministic, kept for API parity).
    """

    def __init__(
        self,
        size: Sequence[int],
        order: int = 1,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.size: tuple[int, ...] = tuple(int(s) for s in size)
        if any(s <= 0 for s in self.size):
            raise ValueError(f"size entries must be > 0, got {size}")
        if order not in (0, 1, 2, 3):
            raise ValueError("order must be 0, 1, 2, or 3")
        self.order = int(order)

    def apply(self, volume: MedVolume) -> MedVolume:
        if len(self.size) != volume.ndim:
            raise ValueError(
                f"size ndim ({len(self.size)}) does not match volume ndim ({volume.ndim})"
            )
        old_shape = volume.shape
        if tuple(self.size) == tuple(old_shape):
            return volume

        factors = [t / s for t, s in zip(self.size, old_shape)]
        image = as_float32(volume.image)
        new_image = zoom(image, factors, order=self.order, mode="reflect")
        # zoom can be off-by-one on rounding; force exact target shape.
        new_image = _force_shape(new_image, self.size, fill=0.0)

        new_mask = None
        if volume.mask is not None:
            m = zoom(volume.mask, factors, order=0, mode="nearest")
            new_mask = _force_shape(m, self.size, fill=0).astype(volume.mask.dtype, copy=False)

        new_spacing = tuple(
            sp * o / t for sp, o, t in zip(volume.spacing, old_shape, self.size)
        )
        point_map = geometry.scale_map(np.asarray(factors, dtype=np.float64))
        return volume.warp(
            point_map,
            image=new_image.astype(np.float32, copy=False),
            mask=new_mask,
            spacing=new_spacing,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {"size": list(self.size), "order": self.order, "p": self.p},
        }


class Pad(Transform):
    """Pad the volume (centred) up to ``size`` along each axis.

    Axes already at or above the target size are left unchanged — ``Pad``
    never crops.  Pair with :class:`CenterCrop` to force an exact shape.

    Args:
        size: Minimum target shape, matching the volume ndim.
        mode: Padding mode forwarded to :func:`numpy.pad` (e.g. ``"constant"``,
            ``"edge"``, ``"reflect"``).
        cval: Constant fill value when ``mode="constant"`` (image only; the
            mask is always padded with ``0``).
        p: Probability of applying.
        seed: RNG seed (unused — Pad is deterministic, kept for API parity).
    """

    def __init__(
        self,
        size: Sequence[int],
        mode: str = "constant",
        cval: float = 0.0,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.size: tuple[int, ...] = tuple(int(s) for s in size)
        if any(s <= 0 for s in self.size):
            raise ValueError(f"size entries must be > 0, got {size}")
        self.mode = str(mode)
        self.cval = float(cval)

    def _pad_width(self, shape: tuple[int, ...]) -> list[tuple[int, int]]:
        widths = []
        for target, cur in zip(self.size, shape):
            extra = max(0, target - cur)
            before = extra // 2
            widths.append((before, extra - before))
        return widths

    def apply(self, volume: MedVolume) -> MedVolume:
        if len(self.size) != volume.ndim:
            raise ValueError(
                f"size ndim ({len(self.size)}) does not match volume ndim ({volume.ndim})"
            )
        widths = self._pad_width(volume.shape)
        if all(b == 0 and a == 0 for b, a in widths):
            return volume

        image = as_float32(volume.image)
        if self.mode == "constant":
            new_image = np.pad(image, widths, mode="constant", constant_values=self.cval)
        else:
            new_image = np.pad(image, widths, mode=self.mode)  # type: ignore[call-overload]

        new_mask = None
        if volume.mask is not None:
            new_mask = np.pad(volume.mask, widths, mode="constant", constant_values=0)

        before = np.asarray([b for b, _ in widths], dtype=np.float64)
        point_map = geometry.translate_map(before)
        return volume.warp(
            point_map, image=new_image.astype(np.float32, copy=False), mask=new_mask
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {
                "size": list(self.size),
                "mode": self.mode,
                "cval": self.cval,
                "p": self.p,
            },
        }


class CenterCrop(Transform):
    """Crop the centred region of ``size`` along each axis.

    Axes smaller than the target size are left unchanged — ``CenterCrop``
    never pads.  Pair with :class:`Pad` to force an exact shape.

    Args:
        size: Target shape, matching the volume ndim.
        p: Probability of applying.
        seed: RNG seed (unused — CenterCrop is deterministic, kept for API parity).
    """

    def __init__(
        self,
        size: Sequence[int],
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.size: tuple[int, ...] = tuple(int(s) for s in size)
        if any(s <= 0 for s in self.size):
            raise ValueError(f"size entries must be > 0, got {size}")

    def apply(self, volume: MedVolume) -> MedVolume:
        if len(self.size) != volume.ndim:
            raise ValueError(
                f"size ndim ({len(self.size)}) does not match volume ndim ({volume.ndim})"
            )
        slices = []
        starts = []
        for target, cur in zip(self.size, volume.shape):
            crop = min(target, cur)
            start = (cur - crop) // 2
            starts.append(start)
            slices.append(slice(start, start + crop))
        region = tuple(slices)
        new_image = volume.image[region].copy()
        new_mask = None if volume.mask is None else volume.mask[region].copy()
        point_map = geometry.translate_map(-np.asarray(starts, dtype=np.float64))
        return volume.warp(point_map, image=new_image, mask=new_mask)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {"size": list(self.size), "p": self.p},
        }


def _force_shape(arr: np.ndarray, target: tuple[int, ...], fill: float) -> np.ndarray:
    """Crop or pad ``arr`` to exactly ``target`` (corrects zoom rounding)."""
    if arr.shape == tuple(target):
        return arr
    slices = tuple(slice(0, min(t, s)) for t, s in zip(target, arr.shape))
    arr = arr[slices]
    if arr.shape != tuple(target):
        pad = [(0, max(0, t - s)) for t, s in zip(target, arr.shape)]
        arr = np.pad(arr, pad, mode="constant", constant_values=fill)
    return arr
