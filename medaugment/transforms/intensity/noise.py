"""Noise models: additive Gaussian and MRI-style Rician."""
from __future__ import annotations

from typing import Union

import numpy as np

from medaugment.core.base import Transform
from medaugment.core.utils import SeedLike, as_float32
from medaugment.core.volume import MedVolume

Range = Union[float, tuple[float, float]]


def _as_std_range(x: Range) -> tuple[float, float]:
    if isinstance(x, (int, float)):
        v = float(x)
        if v < 0:
            raise ValueError(f"std must be >= 0, got {v}")
        return (v, v)
    lo, hi = float(x[0]), float(x[1])
    if lo < 0 or hi < lo:
        raise ValueError(f"std range invalid: {x}")
    return lo, hi


class GaussianNoise(Transform):
    """Additive zero-mean Gaussian noise.

    The standard deviation is interpreted in the **intensity units of the
    image**. For images normalised to ``[0, 1]`` typical values are
    ``0.005–0.05``. For raw HU CT images you'd pass tens to hundreds.

    Args:
        std: Either a fixed value or a ``(low, high)`` range to sample from.
        relative: If True, ``std`` is interpreted as a fraction of the
            current image standard deviation (per-call).
        clip: Optional ``(min, max)`` to clip the output. ``None`` skips clipping.
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        std: Range = 0.01,
        relative: bool = False,
        clip: tuple[float, float] | None = None,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.std_range = _as_std_range(std)
        self.relative = bool(relative)
        self.clip = clip

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)
        std = float(self.rng.uniform(*self.std_range))
        if self.relative:
            std = std * float(image.std() + 1e-8)
        noise = self.rng.normal(loc=0.0, scale=std, size=image.shape).astype(np.float32)
        out = image + noise
        if self.clip is not None:
            out = np.clip(out, self.clip[0], self.clip[1])
        return volume.replace(image=out)


class RicianNoise(Transform):
    """Rician noise — the noise model for magnitude MRI.

    Rician noise arises naturally because the magnitude of complex Gaussian
    noise (which corrupts both real and imaginary k-space channels) is no
    longer Gaussian: ``|N(s, σ) + j·N(0, σ)|`` follows a Rice distribution.
    For high SNR it approaches Gaussian behaviour; for low-signal regions
    (e.g. CSF or air outside the head) it produces the characteristic
    elevated noise floor seen on real MRI.

    Args:
        std: Per-channel sigma — the standard deviation of the underlying
            real/imaginary Gaussians. Either a scalar or ``(low, high)``.
        clip: Optional ``(min, max)`` to clip the result.
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        std: Range = 0.01,
        clip: tuple[float, float] | None = None,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.std_range = _as_std_range(std)
        self.clip = clip

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)
        std = float(self.rng.uniform(*self.std_range))
        real = image + self.rng.normal(0.0, std, size=image.shape).astype(np.float32)
        imag = self.rng.normal(0.0, std, size=image.shape).astype(np.float32)
        out = np.sqrt(real * real + imag * imag).astype(np.float32)
        if self.clip is not None:
            out = np.clip(out, self.clip[0], self.clip[1])
        return volume.replace(image=out)
