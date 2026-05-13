"""Random affine transform (rotation + scaling + translation), 2D and 3D."""
from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy.ndimage import affine_transform

from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike, as_float32
from medaugmentx.core.volume import MedVolume

Range = Union[float, tuple[float, float]]


def _as_range(x: Range, default_centre: float = 0.0) -> tuple[float, float]:
    if isinstance(x, (int, float)):
        v = float(x)
        return (default_centre - v, default_centre + v)
    lo, hi = float(x[0]), float(x[1])
    if lo > hi:
        raise ValueError(f"range lower bound > upper bound: {x}")
    return lo, hi


class RandomAffine(Transform):
    """Random affine transform around the volume centre.

    Supports both 2D and 3D inputs. For 3D, rotation is sampled per axis, and
    you can disable specific axes with ``axes_enabled``. Scaling and
    translation are isotropic-by-default but per-axis ranges may be supplied.

    Anisotropic-aware: passing ``axes_enabled=("x", "y")`` (i.e. disabling
    out-of-plane rotation) is the recommended setup for tomosynthesis volumes
    where the Z axis has very different sampling than XY.

    Args:
        rotation: Maximum rotation in degrees. Either a scalar (interpreted
            as ``(-rotation, rotation)``) or a ``(low, high)`` tuple.
        scale: Per-axis scale factor range. Default ``(1.0, 1.0)`` (no scale).
        translation: Per-axis translation as a fraction of the volume size,
            ``(low, high)``. Default ``(0.0, 0.0)`` (no translation).
        axes_enabled: Which spatial axis labels participate in rotation.
            Defaults to all axes. Use ``("x", "y")`` for in-plane only.
        order: Spline order for image interpolation (0–5). Mask uses ``order=0``.
        mode: ``scipy.ndimage`` boundary mode (``'constant'``, ``'reflect'``,
            ``'nearest'``).
        cval: Fill value for ``mode='constant'``.
        p: Probability of applying the transform.
        seed: RNG seed.
    """

    _LABEL_TO_3D_AXIS = {"z": 0, "y": 1, "x": 2}
    _LABEL_TO_2D_AXIS = {"y": 0, "x": 1}

    def __init__(
        self,
        rotation: Range = 0.0,
        scale: tuple[float, float] = (1.0, 1.0),
        translation: tuple[float, float] = (0.0, 0.0),
        axes_enabled: tuple[str, ...] = ("x", "y", "z"),
        order: int = 1,
        mode: str = "constant",
        cval: float = 0.0,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.rotation_range = _as_range(rotation)
        self.scale_range = (float(scale[0]), float(scale[1]))
        self.translation_range = (float(translation[0]), float(translation[1]))
        self.axes_enabled = tuple(a.lower() for a in axes_enabled)
        if order not in (0, 1, 2, 3, 4, 5):
            raise ValueError("order must be in 0..5")
        self.order = int(order)
        self.mode = mode
        self.cval = float(cval)

    def _sample_params(self, ndim: int) -> dict:
        params: dict = {"rotations_deg": [], "scales": [], "translations": []}
        # Rotations: ndim values (one per axis for 3D). For 2D, only one rotation
        # angle in the plane is meaningful — we use a single value.
        if ndim == 3:
            for label in ("z", "y", "x"):
                if label in self.axes_enabled:
                    params["rotations_deg"].append(
                        float(self.rng.uniform(*self.rotation_range))
                    )
                else:
                    params["rotations_deg"].append(0.0)
        else:  # 2D
            if "x" in self.axes_enabled or "y" in self.axes_enabled:
                params["rotations_deg"].append(
                    float(self.rng.uniform(*self.rotation_range))
                )
            else:
                params["rotations_deg"].append(0.0)
        for _ in range(ndim):
            params["scales"].append(float(self.rng.uniform(*self.scale_range)))
            params["translations"].append(
                float(self.rng.uniform(*self.translation_range))
            )
        return params

    @staticmethod
    def _rotation_matrix_3d(angles_deg: list[float]) -> np.ndarray:
        rz, ry, rx = (np.deg2rad(a) for a in angles_deg)
        cz, sz = np.cos(rz), np.sin(rz)
        cy, sy = np.cos(ry), np.sin(ry)
        cx, sx = np.cos(rx), np.sin(rx)
        # Compose Rz * Ry * Rx; rotations applied around image centre.
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
        return Rz @ Ry @ Rx

    @staticmethod
    def _rotation_matrix_2d(angle_deg: float) -> np.ndarray:
        a = np.deg2rad(angle_deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def _build_matrix_offset(
        self, ndim: int, params: dict, shape: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        if ndim == 3:
            R = self._rotation_matrix_3d(params["rotations_deg"])
        else:
            R = self._rotation_matrix_2d(params["rotations_deg"][0])
        S = np.diag(params["scales"]).astype(np.float64)
        # scipy.ndimage maps output -> input. We want the *forward* transform
        # to be: out = R @ S @ in around the centre. Inverse for the backward
        # warp is therefore (R @ S)^{-1}.
        forward = R @ S
        inverse = np.linalg.inv(forward)
        centre = (np.array(shape, dtype=np.float64) - 1.0) / 2.0
        translation_px = np.array(
            [t * s for t, s in zip(params["translations"], shape)], dtype=np.float64
        )
        # offset such that: input_coord = inverse @ output_coord + offset
        # producing rotation about the centre and the requested translation.
        offset = centre - inverse @ centre - inverse @ translation_px
        return inverse, offset

    def apply(self, volume: MedVolume) -> MedVolume:
        params = self._sample_params(volume.ndim)
        matrix, offset = self._build_matrix_offset(volume.ndim, params, volume.shape)

        image = as_float32(volume.image)
        new_image = affine_transform(
            image,
            matrix=matrix,
            offset=offset,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
            output_shape=image.shape,
            prefilter=self.order > 1,
        )
        new_mask = None
        if volume.mask is not None:
            new_mask = affine_transform(
                volume.mask,
                matrix=matrix,
                offset=offset,
                order=0,
                mode="constant",
                cval=0,
                output_shape=volume.mask.shape,
                prefilter=False,
            ).astype(volume.mask.dtype, copy=False)
        return volume.replace(image=new_image, mask=new_mask)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {
                "rotation": list(self.rotation_range),
                "scale": list(self.scale_range),
                "translation": list(self.translation_range),
                "axes_enabled": list(self.axes_enabled),
                "order": self.order,
                "mode": self.mode,
                "cval": self.cval,
                "p": self.p,
            },
        }
