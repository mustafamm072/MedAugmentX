"""AnisotropicElastic — DBT-tuned elastic deformation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from medaugmentx.core.base import Transform
from medaugmentx.core.parameters import validate_parameters
from medaugmentx.core.utils import SeedLike
from medaugmentx.core.volume import MedVolume
from medaugmentx.transforms.spatial.elastic import ElasticDeform


class AnisotropicElastic(Transform):
    """Elastic deformation with sensible DBT defaults.

    Tomosynthesis voxels are anisotropic (typically ~0.1×0.1mm in-plane vs
    ~1mm slice). Standard isotropic elastic settings warp the slice axis
    far more than is anatomically plausible. This transform is a thin
    wrapper around :class:`~medaugmentx.transforms.spatial.elastic.ElasticDeform`
    with defaults appropriate for DBT, and fails fast on 2D input.

    Defaults in ``(z, y, x)`` order: ``alpha=(8, 100, 100)``,
    ``sigma=(2, 8, 8)``. These are voxel-space settings, not an automatic
    conversion from the volume's spacing.

    Args:
        alpha: Per-axis displacement magnitude in pixels.
        sigma: Per-axis Gaussian sigma.
        order: Image interpolation order (mask uses 0 always).
        p: Probability of applying.
        seed: RNG seed.
        units: ``"voxels"`` or ``"mm"`` for field parameters.
        max_gradient: Optional sampled displacement-gradient norm bound.
        min_jacobian: Optional minimum sampled pull-map Jacobian determinant.
        max_displacement: Optional per-component displacement cap in units.
        bbox_mode: Conservative bounds, explicit corner approximation, or rejection.
    """

    def __init__(
        self,
        alpha: Sequence[float] = (8.0, 100.0, 100.0),
        sigma: Sequence[float] = (2.0, 8.0, 8.0),
        order: int = 1,
        p: float = 1.0,
        seed: SeedLike = None,
        units: str = "voxels",
        max_gradient: float | None = 0.5,
        min_jacobian: float | None = None,
        max_displacement: float | None = None,
        bbox_mode: str = "conservative",
    ) -> None:
        super().__init__(p=p, seed=seed)
        validate_parameters(alpha=alpha, sigma=sigma, order=order)
        self.alpha = tuple(float(a) for a in alpha)
        self.sigma = tuple(float(s) for s in sigma)
        if len(self.alpha) != 3 or len(self.sigma) != 3:
            raise ValueError("AnisotropicElastic expects 3-element alpha/sigma for DBT volumes")
        self._inner = ElasticDeform(
            alpha=self.alpha,
            sigma=self.sigma,
            order=order,
            p=1.0,
            seed=self.rng,
            units=units,
            max_gradient=max_gradient,
            min_jacobian=min_jacobian,
            max_displacement=max_displacement,
            bbox_mode=bbox_mode,
        )

    def set_rng(self, rng: np.random.Generator) -> None:
        super().set_rng(rng)
        # Keep the inner transform's RNG synchronised with ours.
        self._inner.set_rng(rng)

    def apply(self, volume: MedVolume) -> MedVolume:
        if not volume.is_3d:
            raise ValueError("AnisotropicElastic requires a 3D volume")
        return self._inner.apply(volume)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {
                "alpha": list(self.alpha),
                "sigma": list(self.sigma),
                "order": self._inner.order,
                "p": self.p,
                "seed": self._seed,
                "units": self._inner.units,
                "max_gradient": self._inner.max_gradient,
                "min_jacobian": self._inner.min_jacobian,
                "max_displacement": self._inner.max_displacement,
                "bbox_mode": self._inner.bbox_mode,
            },
        }
