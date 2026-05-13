"""LimitedAngleBlur — anisotropic Z-only blur for DBT volumes."""
from __future__ import annotations

from typing import Any, Union

from scipy.ndimage import gaussian_filter1d

from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike, as_float32
from medaugmentx.core.volume import MedVolume

Range = Union[float, tuple[float, float]]


class LimitedAngleBlur(Transform):
    """Z-axis Gaussian blur scaled by the acquisition arc.

    Tomosynthesis reconstructions exhibit out-of-plane blur because the
    detector samples projections over a limited arc (15–25° for typical
    clinical systems). Wider arcs reduce blur (more depth resolution),
    narrower arcs increase it.

    The transform models that effect with a 1-D Gaussian along the slice
    axis. The relationship between sigma (in slices) and the arc angle is
    intentionally approximate — the goal is realistic *variability*, not a
    physics simulator.

    sigma_slices = base_sigma * (reference_arc_deg / arc_degrees)

    Args:
        arc_degrees: Either a fixed arc angle or a ``(low, high)`` range.
        base_sigma: Reference sigma at ``reference_arc_deg``.
        reference_arc_deg: Angle at which sigma equals ``base_sigma``.
        mask_blur_warning: Skip blurring the mask (always); this attribute
            documents the policy.
        p: Probability of applying.
        seed: RNG seed.

    Notes:
        Mask blur would corrupt label boundaries, so the mask is left
        untouched. The image's blur is along Z only — XY is preserved.
    """

    def __init__(
        self,
        arc_degrees: Range = (15.0, 25.0),
        base_sigma: float = 1.0,
        reference_arc_deg: float = 20.0,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if isinstance(arc_degrees, (int, float)):
            self.arc_range: tuple[float, float] = (float(arc_degrees), float(arc_degrees))
        else:
            self.arc_range = (float(arc_degrees[0]), float(arc_degrees[1]))
        if self.arc_range[0] <= 0 or self.arc_range[1] < self.arc_range[0]:
            raise ValueError(f"arc_degrees range invalid: {arc_degrees}")
        self.base_sigma = float(base_sigma)
        if self.base_sigma < 0:
            raise ValueError("base_sigma must be >= 0")
        self.reference_arc_deg = float(reference_arc_deg)
        if self.reference_arc_deg <= 0:
            raise ValueError("reference_arc_deg must be > 0")

    def apply(self, volume: MedVolume) -> MedVolume:
        if not volume.is_3d:
            raise ValueError("LimitedAngleBlur requires a 3D volume")
        arc = float(self.rng.uniform(*self.arc_range))
        sigma = self.base_sigma * (self.reference_arc_deg / arc)
        if sigma <= 0:
            return volume
        image = as_float32(volume.image)
        blurred = gaussian_filter1d(image, sigma=sigma, axis=0, mode="reflect")
        # Mask is intentionally left untouched — see class docstring.
        return volume.replace(image=blurred)

    def to_dict(self) -> dict[str, Any]:
        ar = self.arc_range
        arc_degrees: Any = ar[0] if ar[0] == ar[1] else list(ar)
        return {
            "name": self.__class__.__name__,
            "params": {
                "arc_degrees": arc_degrees,
                "base_sigma": self.base_sigma,
                "reference_arc_deg": self.reference_arc_deg,
                "p": self.p,
            },
        }
