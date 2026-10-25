"""Elastic deformation with anisotropic alpha and sigma."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from medaugmentx.core import geometry
from medaugmentx.core.base import Transform
from medaugmentx.core.parameters import validate_parameters
from medaugmentx.core.utils import SeedLike, as_float32
from medaugmentx.core.volume import MedVolume

ScalarOrSeq = Union[float, Sequence[float]]


def _broadcast(value: ScalarOrSeq, ndim: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return tuple(float(value) for _ in range(ndim))
    out = tuple(float(v) for v in value)
    if len(out) != ndim:
        raise ValueError(f"{name} must be a scalar or have {ndim} entries, got {len(out)}")
    return out


class ElasticDeform(Transform):
    """Elastic deformation via Gaussian-smoothed random displacements.

    Implementation follows the standard Simard / Castro-Pereira approach:

    1. Sample independent uniform displacement fields per spatial axis.
    2. Smooth every field with the same spatial Gaussian sigmas.
    3. Scale each field by axis-specific ``alpha``.
    4. Warp the image (linear interp) and mask (nearest neighbour) using
       :func:`scipy.ndimage.map_coordinates` with the same displacement.

    Anisotropic ``alpha`` and ``sigma`` are essential for tomosynthesis-like
    volumes where the slice axis has very different sampling than the
    in-plane axes (e.g. ``alpha=(10, 120, 120)``, ``sigma=(3, 10, 10)``
    in ``(z, y, x)`` order). Parameters default to voxels. With units="mm",
    values are converted along each grid axis using its spacing; smoothing on
    a sheared grid is separable in grid axes rather than isotropic in RAS.

    Args:
        alpha: Per-axis displacement magnitude in pixels.
        sigma: Gaussian smoothing sigma per spatial axis, applied to every
            displacement component in array-axis order.
        order: Spline order for image interpolation (mask always uses 0).
        mode: Image/mask ``map_coordinates`` boundary mode. For constant
            padding modes, target inversion extends the displacement field
            with its nearest boundary value, independently of image fill.
        cval: Fill value for ``mode='constant'``.
        p: Probability of applying the transform.
        seed: RNG seed.
        units: ``"voxels"`` or ``"mm"`` for alpha, sigma, and displacement cap.
        max_gradient: Maximum sampled displacement-gradient norm; None disables
            automatic limiting. The default reduces extreme draws before warping.
        min_jacobian: Reject sampled pull-map folds below this determinant; None disables.
        max_displacement: Optional per-component absolute displacement cap.
        bbox_mode: ``"conservative"`` bounds using global displacement extrema,
            ``"corners"`` for an explicit approximation, or ``"reject"``.
    """

    def __init__(
        self,
        alpha: ScalarOrSeq = 30.0,
        sigma: ScalarOrSeq = 4.0,
        order: int = 1,
        mode: str = "reflect",
        cval: float = 0.0,
        p: float = 1.0,
        seed: SeedLike = None,
        units: str = "voxels",
        max_gradient: float | None = 0.5,
        min_jacobian: float | None = None,
        max_displacement: float | None = None,
        bbox_mode: str = "conservative",
    ) -> None:
        super().__init__(p=p, seed=seed)
        validate_parameters(alpha=alpha, sigma=sigma, order=order, mode=mode, cval=cval)
        self.alpha_spec = alpha
        self.sigma_spec = sigma
        if order not in (0, 1, 2, 3):
            raise ValueError("order must be 0, 1, 2, or 3 for map_coordinates")
        self.order = int(order)
        self.mode = mode
        self.cval = float(cval)
        validate_parameters(
            alpha=alpha,
            sigma=sigma,
            max_gradient=max_gradient,
            min_jacobian=min_jacobian,
            max_displacement=max_displacement,
        )
        if units not in ("voxels", "mm") or bbox_mode not in ("conservative", "corners", "reject"):
            raise ValueError("Invalid elastic units or bbox_mode")
        if max_gradient is not None and not 0 < max_gradient < 1:
            raise ValueError("max_gradient must be in (0, 1) or None")
        if min_jacobian is not None and min_jacobian <= 0:
            raise ValueError("min_jacobian must be positive or None")
        if max_displacement is not None and max_displacement < 0:
            raise ValueError("max_displacement must be non-negative")
        self.units = units
        self.max_gradient = max_gradient
        self.min_jacobian = min_jacobian
        self.max_displacement = max_displacement
        self.bbox_mode = bbox_mode

    def _displacements(
        self, shape: tuple[int, ...], spacing: tuple[float, ...] | None = None
    ) -> list[np.ndarray]:
        ndim = len(shape)
        alphas = _broadcast(self.alpha_spec, ndim, "alpha")
        sigmas = _broadcast(self.sigma_spec, ndim, "sigma")
        divisor = spacing if self.units == "mm" and spacing is not None else (1.0,) * ndim
        alphas = tuple(a / s for a, s in zip(alphas, divisor))
        sigmas = tuple(a / s for a, s in zip(sigmas, divisor))
        fields = []
        for ax in range(ndim):
            raw = self.rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            if any(s > 0 for s in sigmas):
                raw = gaussian_filter(raw, sigma=sigmas, mode="reflect")
            fields.append(raw * float(alphas[ax]))
        if self.max_displacement is not None:
            fields = [
                np.clip(f, -self.max_displacement / s, self.max_displacement / s)
                for f, s in zip(fields, divisor)
            ]
        if self.max_gradient is not None:
            fields = geometry.limit_displacement_gradient(fields, self.max_gradient)
        if (
            self.min_jacobian is not None
            and (geometry.displacement_jacobian(fields) < self.min_jacobian).any()
        ):
            raise ValueError(
                "Elastic displacement violates min_jacobian; reduce alpha or increase sigma"
            )
        return fields

    def apply(self, volume: MedVolume) -> MedVolume:
        shape = volume.shape
        if volume.bboxes is not None and self.bbox_mode == "reject":
            raise ValueError("Elastic bounding boxes are disabled by bbox_mode=reject")
        displacements = self._displacements(shape, volume.spacing)
        grid = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape], indexing="ij")
        coords = [g + d for g, d in zip(grid, displacements)]
        coord_stack = np.stack([c.ravel() for c in coords], axis=0)

        image = as_float32(volume.image)
        warped = map_coordinates(
            image,
            coord_stack,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
        ).reshape(shape)

        new_mask = None
        if volume.mask is not None:
            warped_mask = (
                map_coordinates(
                    volume.mask,
                    coord_stack,
                    order=0,
                    mode=self.mode,
                    cval=0,
                )
                .reshape(shape)
                .astype(volume.mask.dtype, copy=False)
            )
            new_mask = warped_mask
        # Image fill is not a displacement-field boundary condition: zeroing
        # d outside the grid makes inverse iteration discontinuous at the edge.
        target_mode = "nearest" if self.mode in ("constant", "grid-constant") else self.mode
        point_map = geometry.displacement_map(displacements, mode=target_mode)
        if volume.bboxes is not None and self.bbox_mode == "conservative":
            # q = p - d(q): global component extrema conservatively enclose
            # every mapped interior point, including extrema away from corners.
            low = np.asarray([min(0.0, float(f.min())) for f in displacements])
            high = np.asarray([max(0.0, float(f.max())) for f in displacements])
            return volume.replace(
                image=warped,
                mask=new_mask,
                keypoints=None
                if volume.keypoints is None
                else geometry.map_keypoints(volume.keypoints, point_map),
                bboxes=np.concatenate(
                    [volume.bboxes[:, : volume.ndim] - high, volume.bboxes[:, volume.ndim :] - low],
                    axis=1,
                ),
            )
        return volume.warp(point_map, image=warped, mask=new_mask)

    def to_dict(self) -> dict[str, Any]:
        alpha = (
            list(self.alpha_spec) if isinstance(self.alpha_spec, (list, tuple)) else self.alpha_spec
        )
        sigma = (
            list(self.sigma_spec) if isinstance(self.sigma_spec, (list, tuple)) else self.sigma_spec
        )
        return {
            "name": self.__class__.__name__,
            "params": {
                "alpha": alpha,
                "sigma": sigma,
                "order": self.order,
                "mode": self.mode,
                "cval": self.cval,
                "p": self.p,
                "seed": self._seed,
                "units": self.units,
                "max_gradient": self.max_gradient,
                "min_jacobian": self.min_jacobian,
                "max_displacement": self.max_displacement,
                "bbox_mode": self.bbox_mode,
            },
        }
