"""Axis-aware random flipping."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Union

import numpy as np

from medaugment.core.base import Transform
from medaugment.core.utils import SeedLike, axis_label_to_index
from medaugment.core.volume import MedVolume

AxisSpec = Union[int, str]


class RandomFlip(Transform):
    """Flip the volume along selected axes, each with probability ``p_per_axis``.

    Axes can be given as integer indices (using NumPy ordering, where 3D is
    ``(D, H, W)`` and 2D is ``(H, W)``) or as the labels ``"x"``, ``"y"``,
    ``"z"``. ``"x"`` always means the column axis, ``"y"`` the row axis, and
    ``"z"`` the slice axis (3D only).

    Notes:
        For breast imaging (mammography / DBT) anatomical convention restricts
        valid flips to the horizontal axis: pass ``axes=("x",)``.

    Args:
        axes: Axes to consider for flipping.
        p_per_axis: Independent probability of flipping each enumerated axis,
            evaluated when the transform fires.
        p: Top-level probability that the transform runs at all.
        seed: Optional integer seed or :class:`numpy.random.Generator`.
    """

    def __init__(
        self,
        axes: Iterable[AxisSpec] = ("x",),
        p_per_axis: float = 0.5,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.axes_spec: tuple = tuple(axes)
        if not 0.0 <= float(p_per_axis) <= 1.0:
            raise ValueError(f"p_per_axis must be in [0, 1], got {p_per_axis}")
        self.p_per_axis = float(p_per_axis)

    def _resolve_axes(self, ndim: int) -> Sequence[int]:
        out: list[int] = []
        for a in self.axes_spec:
            if isinstance(a, str):
                out.append(axis_label_to_index(a, ndim))
            else:
                out.append(int(a) if int(a) >= 0 else int(a) + ndim)
        return sorted(set(out))

    def apply(self, volume: MedVolume) -> MedVolume:
        axes = self._resolve_axes(volume.ndim)
        flips = [ax for ax in axes if self.rng.random() < self.p_per_axis]
        if not flips:
            return volume
        new_image = np.flip(volume.image, axis=flips).copy()
        new_mask = None if volume.mask is None else np.flip(volume.mask, axis=flips).copy()
        return volume.replace(image=new_image, mask=new_mask)
