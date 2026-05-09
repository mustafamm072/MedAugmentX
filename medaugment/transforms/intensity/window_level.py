"""Window / level (center / width) perturbation for CT and other modalities."""
from __future__ import annotations

from typing import Any, Union

import numpy as np

from medaugment.core.base import Transform
from medaugment.core.utils import SeedLike, as_float32
from medaugment.core.volume import MedVolume

Range = Union[float, tuple[float, float]]


def _as_range(x: Range, name: str) -> tuple[float, float]:
    if isinstance(x, (int, float)):
        v = float(x)
        return (-v, v)
    lo, hi = float(x[0]), float(x[1])
    if lo > hi:
        raise ValueError(f"{name} lower > upper: {x}")
    return lo, hi


class WindowLevel(Transform):
    """Random window-level augmentation — simulates CT viewer protocol variation.

    Randomly shifts the window centre (by a fraction of the window width)
    and scales the window width, then clips the image to the resulting
    window.  By default the output is linearly rescaled to ``[0, 1]``.

    If the volume's metadata contains ``"window_center"`` and
    ``"window_width"`` keys those are used as the nominal values; otherwise
    the image min/max define the full-range window.

    This is the single highest-impact augmentation for CT models that must
    generalise across different radiological viewing protocols (bone, soft-
    tissue, lung, mediastinum windows).

    Args:
        center_shift_frac: Fractional shift of the window centre relative to
            the window width.  Scalar → symmetric ``(-v, +v)`` range.
        width_scale: Multiplicative scale on the window width.  Scalar →
            fixed; tuple → sampled from ``(lo, hi)``.  Must be > 0.
        rescale_output: If True (default) linearly rescale the clipped
            output to ``[0, 1]``.  Set False to keep the clipped native
            intensity range.
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        center_shift_frac: Range = 0.1,
        width_scale: Range = (0.8, 1.2),
        rescale_output: bool = True,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.center_shift_range = _as_range(center_shift_frac, "center_shift_frac")

        if isinstance(width_scale, (int, float)):
            v = float(width_scale)
            if v <= 0:
                raise ValueError("width_scale must be > 0")
            self.width_scale_range: tuple[float, float] = (v, v)
        else:
            lo, hi = float(width_scale[0]), float(width_scale[1])
            if lo <= 0 or hi < lo:
                raise ValueError(f"width_scale range invalid: {width_scale}")
            self.width_scale_range = (lo, hi)

        self.rescale_output = bool(rescale_output)

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)

        if "window_center" in volume.metadata and "window_width" in volume.metadata:
            center = float(volume.metadata["window_center"])
            width = float(volume.metadata["window_width"])
        else:
            lo = float(image.min())
            hi = float(image.max())
            center = (lo + hi) / 2.0
            width = max(hi - lo, 1e-8)

        center_shift = float(self.rng.uniform(*self.center_shift_range)) * width
        width_scale = float(self.rng.uniform(*self.width_scale_range))

        new_center = center + center_shift
        new_width = max(width * width_scale, 1e-8)
        lo_clip = new_center - new_width / 2.0
        hi_clip = new_center + new_width / 2.0

        out = np.clip(image, lo_clip, hi_clip)
        if self.rescale_output:
            out = (out - lo_clip) / (hi_clip - lo_clip)

        return volume.replace(image=out.astype(np.float32, copy=False))

    def to_dict(self) -> dict[str, Any]:
        csr = self.center_shift_range
        wsr = self.width_scale_range
        # Reconstruct original param: symmetric range → scalar
        center_shift_frac: Any = (-csr[0]) if csr[0] == -csr[1] else list(csr)
        width_scale: Any = wsr[0] if wsr[0] == wsr[1] else list(wsr)
        return {
            "name": self.__class__.__name__,
            "params": {
                "center_shift_frac": center_shift_frac,
                "width_scale": width_scale,
                "rescale_output": self.rescale_output,
                "p": self.p,
            },
        }
