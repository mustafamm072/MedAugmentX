"""Small helpers shared across the library."""
from __future__ import annotations

from typing import Union

import numpy as np

SeedLike = Union[int, np.random.Generator, None]


def resolve_rng(seed: SeedLike) -> np.random.Generator:
    """Return a ``numpy.random.Generator`` from any accepted seed input.

    Accepts ``None`` (fresh entropy), an ``int`` seed, or an existing
    ``Generator`` (returned as-is for chaining).
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def derive_rng(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Spawn ``n`` independent generators from ``rng`` deterministically.

    Used by :class:`~medaugment.core.compose.Compose` to give each child
    transform its own stream while keeping the whole pipeline reproducible
    from a single top-level seed.
    """
    seeds = rng.integers(0, np.iinfo(np.uint64).max, size=n, dtype=np.uint64, endpoint=False)
    return [np.random.default_rng(int(s)) for s in seeds]


def as_float32(image: np.ndarray) -> np.ndarray:
    """Cast to ``float32`` only when needed; cheap no-op otherwise."""
    if image.dtype == np.float32:
        return image
    return image.astype(np.float32, copy=False)


def normalize_axes(axes: int | tuple | list | None, ndim: int) -> tuple:
    """Normalise an ``axes`` argument to a sorted tuple of non-negative ints.

    ``None`` expands to all axes. Negative axes are wrapped relative to ndim.
    """
    if axes is None:
        return tuple(range(ndim))
    if isinstance(axes, int):
        axes = (axes,)
    out: list[int] = []
    for a in axes:
        ax = int(a)
        if ax < 0:
            ax += ndim
        if not 0 <= ax < ndim:
            raise ValueError(f"axis {a} out of range for ndim={ndim}")
        out.append(ax)
    return tuple(sorted(set(out)))


def axis_label_to_index(label: str, ndim: int) -> int:
    """Map a friendly axis label (``"x"``, ``"y"``, ``"z"``) to a NumPy axis.

    Convention used throughout the library:

    - 3D arrays are stored as ``(D, H, W)`` — i.e. ``(z, y, x)``.
    - 2D arrays are stored as ``(H, W)`` — i.e. ``(y, x)``.

    So for 3D ``"z"`` -> 0, ``"y"`` -> 1, ``"x"`` -> 2; for 2D ``"y"`` -> 0,
    ``"x"`` -> 1. ``"z"`` is invalid for 2D arrays.
    """
    label = label.lower()
    if ndim == 3:
        mapping = {"z": 0, "y": 1, "x": 2}
    elif ndim == 2:
        mapping = {"y": 0, "x": 1}
    else:
        raise ValueError(f"Only 2D or 3D supported, got ndim={ndim}")
    if label not in mapping:
        raise ValueError(f"Unknown axis label {label!r} for ndim={ndim}")
    return mapping[label]


def clip_intensity(image: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Clip in-place if writeable, otherwise return a clipped copy."""
    if lo is None and hi is None:
        return image
    return np.clip(image, lo, hi)
