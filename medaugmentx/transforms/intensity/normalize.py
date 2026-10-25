"""Explicit deterministic input normalization for arbitrary-unit modalities."""

from __future__ import annotations

from typing import Any

import numpy as np

from medaugmentx.core import MedVolume, Transform
from medaugmentx.core.parameters import validate_parameters
from medaugmentx.core.utils import SeedLike


class NormalizeIntensity(Transform):
    """Clip to chosen percentiles and map the image into [0, 1].

    Constant finite images become zero. This changes the intensity domain and
    must precede normalized-domain noise; it is inappropriate for preserving HU.

    Args:
        percentiles: Ordered lower/upper percentiles in [0,100].
        p: Probability of applying; use 1 for model input preparation.
        seed: RNG seed, used only for probability gating.
    """

    def __init__(
        self, percentiles: tuple[float, float] = (0.5, 99.5), p: float = 1.0, seed: SeedLike = None
    ) -> None:
        super().__init__(p=p, seed=seed)
        validate_parameters(percentiles=percentiles)
        if len(percentiles) != 2 or not 0 <= percentiles[0] < percentiles[1] <= 100:
            raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
        self.percentiles = tuple(float(v) for v in percentiles)

    def apply(self, volume: MedVolume) -> MedVolume:
        if not np.isfinite(volume.image).all():
            raise ValueError("Normalization requires finite image values")
        image = volume.image.astype(np.float64)
        lo, hi = np.percentile(image, self.percentiles)
        out = np.zeros_like(image) if hi <= lo else (np.clip(image, lo, hi) - lo) / (hi - lo)
        metadata = dict(
            volume.metadata, intensity_domain="normalized", window_center=0.5, window_width=1.0
        )
        return volume.replace(image=out.astype(np.float32), metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": type(self).__name__,
            "params": {"percentiles": self.percentiles, "p": self.p, "seed": self._seed},
        }
