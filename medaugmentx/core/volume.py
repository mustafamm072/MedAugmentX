"""The MedVolume container — image + optional mask + spacing + metadata."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from medaugmentx.core import geometry
from medaugmentx.core.geometry import PointMap


@dataclass
class MedVolume:
    """A single medical image (2D or 3D) with optional segmentation mask.

    All transforms in MedAugmentX operate on this container so that masks,
    geometric targets, and metadata stay in lockstep with the image array.

    Attributes:
        image: 2D ``(H, W)`` or 3D ``(D, H, W)`` array. Recommended dtype is
            ``float32``; integer inputs are accepted but will be cast where
            arithmetic is required.
        mask: Optional integer label map with the same shape as ``image``.
        keypoints: Optional ``(N, ndim)`` float array of landmark coordinates
            in **array-index order** (``(z, y, x)`` for 3D, ``(y, x)`` for 2D —
            the same order used to index ``image``). Spatial transforms warp
            these points in lockstep with the pixels; intensity and artifact
            transforms pass them through unchanged. See
            :mod:`medaugmentx.core.geometry`.
        keypoint_labels: Optional ``(N,)`` array of per-keypoint labels
            (class ids, landmark names, …). Carried through untouched.
        bboxes: Optional ``(M, 2*ndim)`` float array of axis-aligned bounding
            boxes laid out as ``[min…, max…]`` in array-index order. Warped in
            lockstep with the image; under rotation each box becomes the
            axis-aligned bounding box of its transformed corners.
        bbox_labels: Optional ``(M,)`` array of per-box labels. Carried through
            untouched.
        spacing: Voxel size in millimetres, one entry per spatial axis.
            For 3D volumes this is ``(slice_thickness, row_mm, col_mm)``.
        metadata: Free-form dictionary. Conventional keys: ``modality``
            (``"MR" | "CT" | "DX" | "DBT"``), ``vendor``, ``patient_id``,
            ``original_dtype``.
    """

    image: np.ndarray
    mask: np.ndarray | None = None
    spacing: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    # Geometric targets follow spacing/metadata so the original positional
    # signature ``MedVolume(image, mask, spacing, metadata)`` is preserved.
    keypoints: np.ndarray | None = None
    keypoint_labels: np.ndarray | None = None
    bboxes: np.ndarray | None = None
    bbox_labels: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image, np.ndarray):
            raise TypeError(f"image must be a numpy.ndarray, got {type(self.image).__name__}")
        if self.image.ndim not in (2, 3):
            raise ValueError(f"image must be 2D or 3D; got shape {self.image.shape}")

        if self.mask is not None:
            if not isinstance(self.mask, np.ndarray):
                raise TypeError("mask must be a numpy.ndarray or None")
            if self.mask.shape != self.image.shape:
                raise ValueError(
                    f"mask shape {self.mask.shape} does not match image shape {self.image.shape}"
                )

        ndim = int(self.image.ndim)
        self.keypoints, self.keypoint_labels = self._validate_targets(
            self.keypoints, self.keypoint_labels, "keypoint",
            lambda pts: geometry.as_keypoints(pts, ndim),
        )
        self.bboxes, self.bbox_labels = self._validate_targets(
            self.bboxes, self.bbox_labels, "bbox",
            lambda bxs: geometry.as_bboxes(bxs, ndim),
        )

        if self.spacing:
            if len(self.spacing) != self.image.ndim:
                raise ValueError(
                    f"spacing has {len(self.spacing)} entries but image is {self.image.ndim}D"
                )
            self.spacing = tuple(float(s) for s in self.spacing)
        else:
            self.spacing = tuple(1.0 for _ in range(self.image.ndim))

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")

    @staticmethod
    def _validate_targets(
        coords: np.ndarray | None,
        labels: np.ndarray | None,
        name: str,
        coerce: Any,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if coords is None:
            if labels is not None:
                raise ValueError(f"{name}_labels given without {name}s")
            return None, None
        arr = coerce(coords)
        out_labels = None
        if labels is not None:
            out_labels = np.asarray(labels)
            if out_labels.ndim != 1 or out_labels.shape[0] != arr.shape[0]:
                raise ValueError(
                    f"{name}_labels must be 1D with one entry per {name} "
                    f"({arr.shape[0]}); got shape {out_labels.shape}"
                )
        return arr, out_labels

    @property
    def ndim(self) -> int:
        return int(self.image.ndim)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.image.shape)

    @property
    def is_3d(self) -> bool:
        return self.image.ndim == 3

    @property
    def has_mask(self) -> bool:
        return self.mask is not None

    @property
    def has_keypoints(self) -> bool:
        return self.keypoints is not None

    @property
    def has_bboxes(self) -> bool:
        return self.bboxes is not None

    @property
    def num_keypoints(self) -> int:
        return 0 if self.keypoints is None else int(self.keypoints.shape[0])

    @property
    def num_bboxes(self) -> int:
        return 0 if self.bboxes is None else int(self.bboxes.shape[0])

    @property
    def modality(self) -> str | None:
        return self.metadata.get("modality")

    def replace(
        self,
        *,
        image: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        keypoints: np.ndarray | None = None,
        keypoint_labels: np.ndarray | None = None,
        bboxes: np.ndarray | None = None,
        bbox_labels: np.ndarray | None = None,
        spacing: tuple[float, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MedVolume:
        """Return a new MedVolume with selected fields swapped out.

        Any field left as ``None`` keeps its current value, so a transform that
        only rewrites the image (every intensity/artifact transform) implicitly
        carries the mask and geometric targets through unchanged. Metadata is
        shallow-copied to avoid silent aliasing across volumes.
        """
        return replace(
            self,
            image=self.image if image is None else image,
            mask=self.mask if mask is None else mask,
            keypoints=self.keypoints if keypoints is None else keypoints,
            keypoint_labels=(
                self.keypoint_labels if keypoint_labels is None else keypoint_labels
            ),
            bboxes=self.bboxes if bboxes is None else bboxes,
            bbox_labels=self.bbox_labels if bbox_labels is None else bbox_labels,
            spacing=self.spacing if spacing is None else tuple(float(s) for s in spacing),
            metadata=dict(self.metadata if metadata is None else metadata),
        )

    def warp(
        self,
        point_map: PointMap,
        *,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        spacing: tuple[float, ...] | None = None,
    ) -> MedVolume:
        """Return a new volume with a warped image and geometrically mapped targets.

        The single entry point every spatial transform uses: it applies
        ``point_map`` (an ``(K, ndim) -> (K, ndim)`` coordinate map) to the
        keypoints and, via their corners, the bounding boxes, then swaps in the
        already-warped ``image``/``mask``. Labels ride along unchanged. When the
        volume carries no targets this is just :meth:`replace`.

        Args:
            point_map: The forward coordinate map matching the pixel warp — see
                :mod:`medaugmentx.core.geometry` for the standard maps.
            image: The already-transformed image array.
            mask: The already-transformed mask, if any.
            spacing: New voxel spacing (only :class:`~medaugmentx.transforms.Resize`
                needs this); ``None`` keeps the current spacing.
        """
        kp = None if self.keypoints is None else geometry.map_keypoints(self.keypoints, point_map)
        bb = (
            None
            if self.bboxes is None
            else geometry.map_bboxes(self.bboxes, self.ndim, point_map)
        )
        return self.replace(
            image=image, mask=mask, spacing=spacing, keypoints=kp, bboxes=bb
        )

    def remove_out_of_bounds_targets(self, min_visibility: float = 0.0) -> MedVolume:
        """Drop targets that fall outside the current image, in lockstep with labels.

        Spatial transforms map targets faithfully and never drop them, so after
        a crop a keypoint may land at a negative coordinate and a box may hang
        off the edge. Call this to prune such targets for a detection/landmark
        head:

        - keypoints strictly inside ``[0, size)`` on every axis are kept; the
          rest (and their labels) are removed;
        - boxes are clipped to ``[0, size - 1]`` and dropped when their retained
          fraction of the original area is ``< min_visibility`` (with the
          default ``0.0`` only boxes with no overlap at all are dropped).

        Args:
            min_visibility: Minimum fraction (in ``[0, 1]``) of a box's original
                area that must remain inside the frame for it to survive.

        Returns:
            A new :class:`MedVolume` with pruned/clipped targets.
        """
        if not 0.0 <= float(min_visibility) <= 1.0:
            raise ValueError(f"min_visibility must be in [0, 1], got {min_visibility}")
        shape = np.asarray(self.shape, dtype=np.float64)
        new_kp, new_kp_labels = self.keypoints, self.keypoint_labels
        if self.keypoints is not None:
            inside = np.all(
                (self.keypoints >= 0) & (self.keypoints < shape), axis=1
            )
            new_kp = self.keypoints[inside]
            if self.keypoint_labels is not None:
                new_kp_labels = self.keypoint_labels[inside]

        new_bb, new_bb_labels = self.bboxes, self.bbox_labels
        if self.bboxes is not None:
            new_bb, keep = self._clip_bboxes(self.bboxes, shape, float(min_visibility))
            if self.bbox_labels is not None:
                new_bb_labels = self.bbox_labels[keep]

        out = self.copy()
        out.keypoints, out.keypoint_labels = new_kp, new_kp_labels
        out.bboxes, out.bbox_labels = new_bb, new_bb_labels
        return out

    @staticmethod
    def _clip_bboxes(
        bboxes: np.ndarray, shape: np.ndarray, min_visibility: float
    ) -> tuple[np.ndarray, np.ndarray]:
        ndim = shape.shape[0]
        mins, maxs = bboxes[:, :ndim], bboxes[:, ndim:]
        original = np.prod(np.clip(maxs - mins, 0.0, None), axis=1)
        hi = shape - 1.0
        cmins = np.clip(mins, 0.0, hi)
        cmaxs = np.clip(maxs, 0.0, hi)
        clipped_area = np.prod(np.clip(cmaxs - cmins, 0.0, None), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(original > 0, clipped_area / original, 0.0)
        keep = (clipped_area > 0) & (frac >= min_visibility)
        clipped = np.concatenate([cmins, cmaxs], axis=1)[keep]
        return clipped, keep

    def copy(self) -> MedVolume:
        """Deep copy of the underlying arrays and metadata."""
        return MedVolume(
            image=self.image.copy(),
            mask=None if self.mask is None else self.mask.copy(),
            keypoints=None if self.keypoints is None else self.keypoints.copy(),
            keypoint_labels=(
                None if self.keypoint_labels is None else self.keypoint_labels.copy()
            ),
            bboxes=None if self.bboxes is None else self.bboxes.copy(),
            bbox_labels=None if self.bbox_labels is None else self.bbox_labels.copy(),
            spacing=tuple(self.spacing),
            metadata=dict(self.metadata),
        )

    def __repr__(self) -> str:
        mask_repr = "None" if self.mask is None else f"shape={self.mask.shape}, dtype={self.mask.dtype}"
        targets = ""
        if self.keypoints is not None:
            targets += f", keypoints={self.num_keypoints}"
        if self.bboxes is not None:
            targets += f", bboxes={self.num_bboxes}"
        return (
            f"MedVolume(image=shape={self.image.shape}, dtype={self.image.dtype}, "
            f"mask={mask_repr}{targets}, spacing={self.spacing}, "
            f"modality={self.modality!r})"
        )
