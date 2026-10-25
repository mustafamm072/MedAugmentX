"""Shared constructor validation for finite values, sizes, ranges, and modes."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np

NDIMAGE_MODES = {
    "reflect",
    "grid-mirror",
    "constant",
    "grid-constant",
    "nearest",
    "mirror",
    "grid-wrap",
    "wrap",
}


def finite_values(value: Any, name: str) -> None:
    """Reject nonfinite numeric constructor values, including nested references."""
    if isinstance(value, Mapping):
        for child in value.values():
            finite_values(child, name)
    elif isinstance(value, np.ndarray):
        if value.dtype.kind == "c":
            raise ValueError(f"{name} must contain real values")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite_values(child, name)
    elif isinstance(value, (complex, np.complexfloating)):
        raise ValueError(f"{name} must contain real values")
    elif isinstance(value, (Real, np.number)) and not np.isfinite(float(value)):
        raise ValueError(f"{name} must contain only finite values")


def positive_size(value: Any, name: str = "size") -> tuple[int, ...]:
    values = (value,) if isinstance(value, Integral) else tuple(value)
    if not values or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v <= 0 for v in values
    ):
        raise ValueError(f"{name} must contain positive integers")
    return tuple(int(v) for v in values)


def validate_parameters(**parameters: Any) -> None:
    """Common checks invoked explicitly by built-in transform constructors.

    Effect-specific ordering and dimensionality checks remain in each class.
    This helper does not alter user-defined transforms or constructor signatures.
    """
    integer_names = {
        "num_holes",
        "num_slices",
        "num_movements",
        "num_sources",
        "num_streaks",
        "num_planes",
        "num_ghosts",
        "n_bins",
        "n_quantiles",
        "max_shift",
        "ghost_shift",
        "ksize",
        "order",
        "order_up",
        "order_down",
    }
    positive_names = {
        "gamma",
        "contrast",
        "scale",
        "width_scale",
        "falloff",
        "power",
        "arc_degrees",
        "reference_arc_deg",
        "clip_limit",
        "frequency",
    }
    nonnegative_names = {
        "alpha",
        "sigma",
        "std",
        "base_sigma",
        "intensity",
        "amplitude",
        "ghost_intensity",
        "fraction",
    }
    range_names = {
        "rotation",
        "scale",
        "translation",
        "brightness",
        "contrast",
        "gamma",
        "std",
        "center_shift_frac",
        "width_scale",
        "dropout_fraction",
        "ghost_shift",
        "ghost_intensity",
        "num_holes",
        "hole_size",
        "num_slices",
        "max_shift",
        "arc_degrees",
        "zoom_range",
        "clip_limit",
        "blend",
        "intensity",
        "num_streaks",
        "num_sources",
        "num_planes",
        "amplitude",
        "num_movements",
        "degrees",
        "frequency",
        "fraction",
        "ksize",
    }
    for name, value in parameters.items():
        if value is None or name == "seed":
            continue
        finite_values(value, name)
        if name in range_names and isinstance(value, (tuple, list)):
            if len(value) != 2 or value[0] > value[1]:
                raise ValueError(f"{name} must be an ordered two-element range")
        if name in {"size", "coarse_shape", "grid"}:
            positive_size(value, name)
        values = value if isinstance(value, (list, tuple)) else (value,)
        if name in integer_names:
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in values):
                raise ValueError(f"{name} must contain integers")
        if name in positive_names and any(v <= 0 for v in values):
            raise ValueError(f"{name} must be positive")
        if name in nonnegative_names and any(v < 0 for v in values):
            raise ValueError(f"{name} must be non-negative")
        if (
            name in {"rotation", "degrees", "brightness", "center_shift_frac", "max_shift"}
            and isinstance(value, Real)
            and value < 0
        ):
            raise ValueError(f"{name} scalar magnitude must be non-negative")
        if name == "clip" and (len(value) != 2 or value[0] > value[1]):
            raise ValueError("clip must be an ordered (low, high) pair")
        if name == "mode" and value not in NDIMAGE_MODES and parameters.get("_padding") is not True:
            raise ValueError(f"Unsupported boundary mode: {value!r}")


def checked_range(value: Any, name: str, *, minimum: float | None = None) -> tuple[float, float]:
    finite_values(value, name)
    if isinstance(value, Real):
        values = (float(value), float(value))
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must be a scalar or two-element range")
        values = (float(value[0]), float(value[1]))
    if values[0] > values[1] or (minimum is not None and values[0] < minimum):
        raise ValueError(f"{name} must be an ordered range with lower bound >= {minimum}")
    return values
