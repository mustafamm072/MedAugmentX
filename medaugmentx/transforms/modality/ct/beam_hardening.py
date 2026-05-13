"""CT beam hardening (cupping) artifact simulation."""
from __future__ import annotations

from typing import Any, Union

import numpy as np

from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike, as_float32
from medaugmentx.core.volume import MedVolume

Range = Union[float, tuple[float, float]]


class BeamHardening(Transform):
    """Simulate CT beam hardening (cupping) artifact.

    When a polychromatic X-ray beam passes through tissue, lower-energy
    photons are preferentially absorbed (photoelectric absorption dominates).
    The beam reaching the centre of a homogeneous object is richer in
    high-energy photons, which scatter less and thus appear *less* attenuating
    than expected — the reconstructed centre looks darker than the periphery.
    This is the *cupping artifact*.

    Implementation: subtract a smooth radially-symmetric bowl profile from
    each axial slice, scaled by ``alpha * image_range``.  The bowl peaks at
    the centre (maximal darkening) and tapers to zero at the edges.

    ``out = image - alpha * bowl(H, W) * image_range``

    Works on 2-D and 3-D inputs.  For 3-D, the same bowl is applied to every
    axial slice.  The mask is never modified.

    Args:
        alpha: Artifact strength relative to the image intensity range.
            Scalar → fixed; tuple → sampled from ``(lo, hi)``.
            Typical values: ``0.02–0.10``.
        power: Exponent controlling the bowl shape (higher → sharper cupping
            restricted to the very centre; lower → broader dimming).
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        alpha: Range = (0.02, 0.08),
        power: float = 2.0,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if isinstance(alpha, (int, float)):
            v = float(alpha)
            if v < 0:
                raise ValueError(f"alpha must be >= 0, got {alpha}")
            self.alpha_range: tuple[float, float] = (v, v)
        else:
            lo, hi = float(alpha[0]), float(alpha[1])
            if lo < 0 or hi < lo:
                raise ValueError(f"alpha range invalid: {alpha}")
            self.alpha_range = (lo, hi)
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}")
        self.power = float(power)

    def _make_bowl(self, h: int, w: int) -> np.ndarray:
        """Return a (H, W) radial bowl: 1.0 at centre, 0.0 at boundary."""
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        y = np.linspace(0, h - 1, h, dtype=np.float32)
        x = np.linspace(0, w - 1, w, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        # Normalised radius: 0 at centre, 1 at corner of bounding circle
        r = np.sqrt(
            ((yy - cy) / max(cy, 1.0)) ** 2 + ((xx - cx) / max(cx, 1.0)) ** 2
        )
        r = np.clip(r, 0.0, 1.0)
        return (1.0 - r**self.power).astype(np.float32)

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)
        alpha = float(self.rng.uniform(*self.alpha_range))

        img_range = float(image.max()) - float(image.min())
        if img_range < 1e-8:
            return volume

        h, w = image.shape[-2], image.shape[-1]
        bowl = self._make_bowl(h, w)  # (H, W)
        correction = alpha * bowl * img_range  # darkens the centre

        if volume.is_3d:
            out = image - correction[np.newaxis, :, :]
        else:
            out = image - correction

        return volume.replace(image=out.astype(np.float32, copy=False))

    def to_dict(self) -> dict[str, Any]:
        ar = self.alpha_range
        return {
            "name": self.__class__.__name__,
            "params": {
                "alpha": ar[0] if ar[0] == ar[1] else list(ar),
                "power": self.power,
                "p": self.p,
            },
        }
