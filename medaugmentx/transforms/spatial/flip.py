"""Axis-aware random flipping."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Union

import numpy as np

from medaugmentx.core import geometry
from medaugmentx.core.base import Transform
from medaugmentx.core.parameters import validate_parameters
from medaugmentx.core.utils import SeedLike, axis_label_to_index, normalize_axes
from medaugmentx.core.volume import MedVolume

AxisSpec = Union[int, str]


class RandomFlip(Transform):
    """Flip the volume along selected axes, each with probability ``p_per_axis``.

    Axes can be given as integer indices (using NumPy ordering, where 3D is
    ``(D, H, W)`` and 2D is ``(H, W)``) or as the labels ``"x"``, ``"y"``,
    ``"z"``. ``"x"`` always means the column axis, ``"y"`` the row axis, and
    ``"z"`` the slice axis (3D only).

    Notes:
        Task and acquisition conventions determine whether any reflection is
        valid. Array x does not establish anatomical laterality. Presets leave
        flips disabled unless explicitly enabled.

    Args:
        axes: Axes to consider for flipping.
        p_per_axis: Independent probability of flipping each enumerated axis,
            evaluated when the transform fires.
        p: Top-level probability that the transform runs at all.
        seed: Optional integer seed or :class:`numpy.random.Generator`.
        label_map: Optional involutive segmentation class mapping on label_axis flips.
        keypoint_label_map: Optional involutive landmark-name/class mapping.
        bbox_label_map: Optional involutive box-class mapping.
        label_axis: Array axis whose flip triggers label remapping, default x.
    """

    def __init__(
        self,
        axes: Iterable[AxisSpec] = ("x",),
        p_per_axis: float = 0.5,
        p: float = 1.0,
        seed: SeedLike = None,
        label_map: dict[int, int] | None = None,
        keypoint_label_map: dict | None = None,
        bbox_label_map: dict | None = None,
        label_axis: str = "x",
    ) -> None:
        super().__init__(p=p, seed=seed)
        validate_parameters(axes=axes, p_per_axis=p_per_axis)
        self.axes_spec: tuple = tuple(axes)
        if not 0.0 <= float(p_per_axis) <= 1.0:
            raise ValueError(f"p_per_axis must be in [0, 1], got {p_per_axis}")
        self.p_per_axis = float(p_per_axis)
        self.label_map = self._check_map(label_map)
        self.keypoint_label_map = self._check_map(keypoint_label_map)
        self.bbox_label_map = self._check_map(bbox_label_map)
        if label_axis not in ("x", "y", "z"):
            raise ValueError("label_axis must be x, y, or z")
        self.label_axis = label_axis

    @staticmethod
    def _check_map(mapping):
        mapping = None if mapping is None else dict(mapping)
        if mapping is not None and any(
            mapping.get(target) != source for source, target in mapping.items()
        ):
            raise ValueError(
                "Flip label maps must be involutions (each swapped pair maps both ways)"
            )
        return None if mapping is None else dict(mapping)

    @staticmethod
    def _remap(labels, mapping):
        if labels is None or mapping is None:
            return labels
        return np.asarray([mapping.get(value, value) for value in labels.tolist()])

    def _resolve_axes(self, ndim: int) -> Sequence[int]:
        out: list[int] = []
        for a in self.axes_spec:
            if isinstance(a, str):
                out.append(axis_label_to_index(a, ndim))
            else:
                if not isinstance(a, (int, np.integer)) or isinstance(a, (bool, np.bool_)):
                    raise ValueError("Flip axes must be labels or integer indices")
                out.append(int(a) if int(a) >= 0 else int(a) + ndim)
        return normalize_axes(out, ndim)

    def apply(self, volume: MedVolume) -> MedVolume:
        axes = self._resolve_axes(volume.ndim)
        flips = [ax for ax in axes if self.rng.random() < self.p_per_axis]
        if not flips:
            return volume
        flip_axes = tuple(flips)
        new_image = np.flip(volume.image, axis=flip_axes).copy()
        new_mask = None if volume.mask is None else np.flip(volume.mask, axis=flip_axes).copy()
        point_map = geometry.flip_map(flip_axes, volume.shape)
        out = volume.warp(point_map, image=new_image, mask=new_mask)
        if axis_label_to_index(self.label_axis, volume.ndim) in flip_axes:
            if new_mask is not None and self.label_map is not None:
                info = np.iinfo(new_mask.dtype) if new_mask.dtype.kind in "iu" else None
                for source, target in self.label_map.items():
                    if not isinstance(source, (int, np.integer)) or not isinstance(
                        target, (int, np.integer)
                    ):
                        raise ValueError("Segmentation label maps must use integer class IDs")
                    if info is not None and not info.min <= target <= info.max:
                        raise ValueError(
                            "Remapped class ID does not fit mask dtype; cast mask first"
                        )
                original = new_mask.copy()
                for source, target in self.label_map.items():
                    new_mask[original == source] = target
            out = out.replace(
                mask=new_mask,
                keypoint_labels=self._remap(out.keypoint_labels, self.keypoint_label_map),
                bbox_labels=self._remap(out.bbox_labels, self.bbox_label_map),
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {
                "axes": list(self.axes_spec),
                "p_per_axis": self.p_per_axis,
                "p": self.p,
                "seed": self._seed,
                **(
                    {"label_map": list(self.label_map.items())}
                    if self.label_map is not None
                    else {}
                ),
                **(
                    {"keypoint_label_map": list(self.keypoint_label_map.items())}
                    if self.keypoint_label_map is not None
                    else {}
                ),
                **(
                    {"bbox_label_map": list(self.bbox_label_map.items())}
                    if self.bbox_label_map is not None
                    else {}
                ),
                **(
                    {"label_axis": self.label_axis}
                    if any(
                        m is not None
                        for m in (self.label_map, self.keypoint_label_map, self.bbox_label_map)
                    )
                    or self.label_axis != "x"
                    else {}
                ),
            },
        }
