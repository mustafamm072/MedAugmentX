"""AnisotropicElastic — DBT-tuned elastic deformation."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from medaugmentx.core.base import Transform
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

    Defaults: ``alpha=(100, 100, 8)``, ``sigma=(8, 8, 2)``.

    Args:
        alpha: Per-axis displacement magnitude in pixels.
        sigma: Per-axis Gaussian sigma.
        order: Image interpolation order (mask uses 0 always).
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        alpha: Sequence[float] = (100.0, 100.0, 8.0),
        sigma: Sequence[float] = (8.0, 8.0, 2.0),
        order: int = 1,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if len(tuple(alpha)) != 3 or len(tuple(sigma)) != 3:
            raise ValueError("AnisotropicElastic expects 3-element alpha/sigma for DBT volumes")
        self._inner = ElasticDeform(
            alpha=tuple(alpha),
            sigma=tuple(sigma),
            order=order,
            p=1.0,
            seed=self.rng,
        )

    def set_rng(self, rng) -> None:  # type: ignore[override]
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
                "alpha": list(self._inner.alpha_spec),
                "sigma": list(self._inner.sigma_spec),
                "order": self._inner.order,
                "p": self.p,
            },
        }
