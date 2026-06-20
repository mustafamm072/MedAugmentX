"""Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
from __future__ import annotations

from typing import Any, Union

import numpy as np

from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike, as_float32
from medaugmentx.core.volume import MedVolume

Range = Union[float, tuple[float, float]]


def _clahe_2d(img01: np.ndarray, n_bins: int, gy: int, gx: int, clip_limit: float) -> np.ndarray:
    """CLAHE on a single 2-D image already scaled to ``[0, 1]``.

    Tiles the image into a ``gy x gx`` grid, builds a clipped, contrast-limited
    histogram-equalisation mapping per tile, and bilinearly interpolates the
    per-tile mappings across the image to avoid blocky tile boundaries.
    """
    h, w = img01.shape
    th = int(np.ceil(h / gy))
    tw = int(np.ceil(w / gx))
    hp, wp = th * gy, tw * gx
    img = np.pad(img01, ((0, hp - h), (0, wp - w)), mode="edge")

    q = np.clip(np.rint(img * (n_bins - 1)).astype(np.int64), 0, n_bins - 1)

    tile_pixels = th * tw
    clip_count = max(1, int(clip_limit * tile_pixels / n_bins))

    luts = np.empty((gy, gx, n_bins), dtype=np.float64)
    for i in range(gy):
        for j in range(gx):
            tile = q[i * th : (i + 1) * th, j * tw : (j + 1) * tw]
            hist = np.bincount(tile.ravel(), minlength=n_bins).astype(np.float64)
            excess = np.maximum(hist - clip_count, 0.0).sum()
            hist = np.minimum(hist, clip_count) + excess / n_bins
            cdf = np.cumsum(hist)
            total = cdf[-1]
            luts[i, j] = cdf / total if total > 0 else cdf

    # Fractional tile coordinate of each row/column, clamped to the centres.
    fy = np.clip((np.arange(hp) - (th / 2 - 0.5)) / th, 0, gy - 1)
    fx = np.clip((np.arange(wp) - (tw / 2 - 0.5)) / tw, 0, gx - 1)
    iy0 = np.floor(fy).astype(np.int64)
    ix0 = np.floor(fx).astype(np.int64)
    iy1 = np.minimum(iy0 + 1, gy - 1)
    ix1 = np.minimum(ix0 + 1, gx - 1)
    wy = (fy - iy0)[:, None]
    wx = (fx - ix0)[None, :]

    IY0, IX0 = iy0[:, None] + np.zeros_like(ix0)[None, :], ix0[None, :] + np.zeros_like(iy0)[:, None]
    IY1 = iy1[:, None] + np.zeros_like(ix0)[None, :]
    IX1 = ix1[None, :] + np.zeros_like(iy0)[:, None]

    v00 = luts[IY0, IX0, q]
    v01 = luts[IY0, IX1, q]
    v10 = luts[IY1, IX0, q]
    v11 = luts[IY1, IX1, q]
    out = (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11)
    return out[:h, :w]


class CLAHEContrast(Transform):
    """Contrast Limited Adaptive Histogram Equalization.

    CLAHE equalises contrast locally rather than globally, revealing detail
    in both dark and bright regions without amplifying noise the way plain
    histogram equalisation does (the per-tile histogram is *clipped* before
    the mapping is built).  It is one of the most widely used pre-processing /
    augmentation steps in radiology, especially for chest X-ray and mammography.

    The image is rescaled to ``[0, 1]`` internally and mapped back to its
    original intensity range on output.  For 3-D volumes CLAHE is applied
    slice-by-slice along the first (Z) axis.  The mask is never modified.

    Args:
        clip_limit: Contrast-clip threshold relative to the per-tile mean
            histogram height.  Higher → more contrast (and more noise).
            Scalar → fixed; tuple ``(lo, hi)`` → sampled per call.  Typical
            values ``1.0–4.0``.
        grid: Number of tiles per axis as ``(rows, cols)`` for the 2-D
            equalisation grid (applied in-plane).
        n_bins: Number of histogram bins (256 by default).
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        clip_limit: Range = (1.0, 3.0),
        grid: tuple[int, int] = (8, 8),
        n_bins: int = 256,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if isinstance(clip_limit, (int, float)):
            v = float(clip_limit)
            if v <= 0:
                raise ValueError("clip_limit must be > 0")
            self.clip_range: tuple[float, float] = (v, v)
        else:
            lo, hi = float(clip_limit[0]), float(clip_limit[1])
            if lo <= 0 or hi < lo:
                raise ValueError(f"clip_limit range invalid: {clip_limit}")
            self.clip_range = (lo, hi)

        gy, gx = int(grid[0]), int(grid[1])
        if gy < 1 or gx < 1:
            raise ValueError(f"grid entries must be >= 1, got {grid}")
        self.grid: tuple[int, int] = (gy, gx)

        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.n_bins = int(n_bins)

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)
        lo = float(image.min())
        hi = float(image.max())
        if hi - lo < 1e-12:
            return volume  # constant image — CLAHE is identity

        clip_limit = float(self.rng.uniform(*self.clip_range))
        img01 = (image - lo) / (hi - lo)
        gy, gx = self.grid

        if volume.is_3d:
            out01 = np.empty_like(img01)
            for z in range(img01.shape[0]):
                out01[z] = _clahe_2d(img01[z], self.n_bins, gy, gx, clip_limit)
        else:
            out01 = _clahe_2d(img01, self.n_bins, gy, gx, clip_limit)

        out = out01 * (hi - lo) + lo
        return volume.replace(image=out.astype(np.float32, copy=False))

    def to_dict(self) -> dict[str, Any]:
        cr = self.clip_range
        return {
            "name": self.__class__.__name__,
            "params": {
                "clip_limit": cr[0] if cr[0] == cr[1] else list(cr),
                "grid": list(self.grid),
                "n_bins": self.n_bins,
                "p": self.p,
            },
        }
