"""Histogram matching to a reference intensity distribution."""
from __future__ import annotations

from typing import Any, Union

import numpy as np

from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike, as_float32
from medaugmentx.core.volume import MedVolume

Range = Union[float, tuple[float, float]]


def _match_to_reference(image: np.ndarray, ref_values: np.ndarray, ref_quantiles: np.ndarray) -> np.ndarray:
    """Map ``image`` so its CDF matches the reference (values, quantiles)."""
    flat = image.ravel()
    src_values, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    src_quantiles = np.cumsum(counts).astype(np.float64) / flat.size
    # For each distinct source value, find the reference intensity at the same quantile.
    interp_values = np.interp(src_quantiles, ref_quantiles, ref_values)
    return interp_values[inverse].reshape(image.shape)


class HistogramMatch(Transform):
    """Match the image intensity histogram to a reference distribution.

    Standardising intensity profiles across scanners and acquisition
    protocols is a common harmonisation step for multi-centre studies; used
    as an augmentation it teaches a model to be robust to that variation by
    re-shaping each volume toward a fixed reference distribution.

    Provide one reference image (any shape) whose intensity *distribution* is
    the target.  The output preserves the input shape; only voxel intensities
    are remapped.  The mask is never modified.

    .. note::
        The reference array is serialised inline (as a nested list).  Keep it
        small — a representative slice or a downsampled volume is plenty.  If
        ``reference`` is ``None`` the transform is an identity no-op, which
        lets a pipeline round-trip cleanly when you do not want to persist a
        reference array.

    Args:
        reference: Array-like whose histogram is the matching target, or
            ``None`` for an identity no-op.
        blend: Blend ratio between the original and matched image, in
            ``[0, 1]``.  ``1`` = full match (default), ``0`` = identity.
            Scalar → fixed; tuple ``(lo, hi)`` → sampled per call.
        n_quantiles: Number of quantiles used to summarise the reference CDF.
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        reference: Any = None,
        blend: Range = 1.0,
        n_quantiles: int = 256,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if isinstance(blend, (int, float)):
            v = float(blend)
            if not 0.0 <= v <= 1.0:
                raise ValueError("blend must be in [0, 1]")
            self.blend_range: tuple[float, float] = (v, v)
        else:
            lo, hi = float(blend[0]), float(blend[1])
            if not (0.0 <= lo <= hi <= 1.0):
                raise ValueError(f"blend range must satisfy 0 <= lo <= hi <= 1, got {blend}")
            self.blend_range = (lo, hi)

        if n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")
        self.n_quantiles = int(n_quantiles)

        if reference is None:
            self.reference: np.ndarray | None = None
            self._ref_values: np.ndarray | None = None
            self._ref_quantiles: np.ndarray | None = None
        else:
            ref = np.asarray(reference, dtype=np.float64)
            self.reference = ref
            qs = np.linspace(0.0, 1.0, self.n_quantiles)
            self._ref_values = np.quantile(ref.ravel(), qs)
            self._ref_quantiles = qs

    def apply(self, volume: MedVolume) -> MedVolume:
        if self.reference is None or self._ref_values is None or self._ref_quantiles is None:
            return volume
        image = as_float32(volume.image)
        matched = _match_to_reference(
            image.astype(np.float64), self._ref_values, self._ref_quantiles
        )
        blend = float(self.rng.uniform(*self.blend_range))
        out = (1.0 - blend) * image + blend * matched
        return volume.replace(image=out.astype(np.float32, copy=False))

    def to_dict(self) -> dict[str, Any]:
        br = self.blend_range
        return {
            "name": self.__class__.__name__,
            "params": {
                "reference": None if self.reference is None else self.reference.tolist(),
                "blend": br[0] if br[0] == br[1] else list(br),
                "n_quantiles": self.n_quantiles,
                "p": self.p,
            },
        }
