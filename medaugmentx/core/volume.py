"""The MedVolume container — image + optional mask + spacing + metadata."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


@dataclass
class MedVolume:
    """A single medical image (2D or 3D) with optional segmentation mask.

    All transforms in MedAugment operate on this container so that masks and
    metadata stay in lockstep with the image array.

    Attributes:
        image: 2D ``(H, W)`` or 3D ``(D, H, W)`` array. Recommended dtype is
            ``float32``; integer inputs are accepted but will be cast where
            arithmetic is required.
        mask: Optional integer label map with the same shape as ``image``.
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
    def modality(self) -> str | None:
        return self.metadata.get("modality")

    def replace(
        self,
        *,
        image: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        spacing: tuple[float, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MedVolume:
        """Return a new MedVolume with selected fields swapped out.

        Use ``mask=...`` only to provide a new mask; pass ``mask=None`` and
        rely on the existing one by omitting the keyword. Metadata is shallow-
        copied to avoid silent aliasing across volumes.
        """
        return replace(
            self,
            image=self.image if image is None else image,
            mask=self.mask if mask is None else mask,
            spacing=self.spacing if spacing is None else tuple(float(s) for s in spacing),
            metadata=dict(self.metadata if metadata is None else metadata),
        )

    def copy(self) -> MedVolume:
        """Deep copy of the underlying arrays and metadata."""
        return MedVolume(
            image=self.image.copy(),
            mask=None if self.mask is None else self.mask.copy(),
            spacing=tuple(self.spacing),
            metadata=dict(self.metadata),
        )

    def __repr__(self) -> str:
        mask_repr = "None" if self.mask is None else f"shape={self.mask.shape}, dtype={self.mask.dtype}"
        return (
            f"MedVolume(image=shape={self.image.shape}, dtype={self.image.dtype}, "
            f"mask={mask_repr}, spacing={self.spacing}, "
            f"modality={self.modality!r})"
        )
