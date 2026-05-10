"""Pipeline builders: Compose, OneOf, SomeOf."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from medaugment.core.base import Transform
from medaugment.core.utils import SeedLike, derive_rng
from medaugment.core.volume import MedVolume


class Compose(Transform):
    """Apply transforms sequentially.

    All children share a deterministic seeding chain derived from the
    top-level seed, so ``Compose([...], seed=42)`` produces the same output
    every time, on every machine, for the same NumPy version.
    """

    def __init__(
        self,
        transforms: Iterable[Transform],
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.transforms: list[Transform] = list(transforms)
        for t in self.transforms:
            if not isinstance(t, Transform):
                raise TypeError(
                    f"Compose expected Transform instances, got {type(t).__name__}"
                )
        self._reseed_children()

    def _reseed_children(self) -> None:
        if not self.transforms:
            return
        for t, child_rng in zip(self.transforms, derive_rng(self.rng, len(self.transforms))):
            t.set_rng(child_rng)

    def set_rng(self, rng: np.random.Generator) -> None:
        super().set_rng(rng)
        self._reseed_children()

    def apply(self, volume: MedVolume) -> MedVolume:
        out = volume
        for t in self.transforms:
            out = t(out)
        return out

    def __len__(self) -> int:
        return len(self.transforms)

    def __iter__(self):
        return iter(self.transforms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {
                "transforms": [t.to_dict() for t in self.transforms],
                "p": self.p,
                "seed": self._seed,
            },
        }


class OneOf(Transform):
    """Pick exactly one child uniformly at random and apply it.

    The container's ``p`` controls whether *any* child runs at all. When
    weights are provided they are normalised; otherwise the choice is uniform.
    """

    def __init__(
        self,
        transforms: Sequence[Transform],
        weights: Sequence[float] | None = None,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.transforms: list[Transform] = list(transforms)
        if not self.transforms:
            raise ValueError("OneOf requires at least one transform")
        for t in self.transforms:
            if not isinstance(t, Transform):
                raise TypeError(f"OneOf expected Transform, got {type(t).__name__}")

        if weights is None:
            self.weights = np.full(len(self.transforms), 1.0 / len(self.transforms))
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (len(self.transforms),):
                raise ValueError("weights length must match number of transforms")
            if (w < 0).any() or w.sum() <= 0:
                raise ValueError("weights must be non-negative and sum to > 0")
            self.weights = w / w.sum()

        self._reseed_children()

    def _reseed_children(self) -> None:
        for t, child_rng in zip(self.transforms, derive_rng(self.rng, len(self.transforms))):
            t.set_rng(child_rng)

    def set_rng(self, rng: np.random.Generator) -> None:
        super().set_rng(rng)
        self._reseed_children()

    def apply(self, volume: MedVolume) -> MedVolume:
        idx = int(self.rng.choice(len(self.transforms), p=self.weights))
        # Force the chosen child to run regardless of its own ``p``.
        return self.transforms[idx].apply(volume)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {
                "transforms": [t.to_dict() for t in self.transforms],
                "weights": self.weights.tolist(),
                "p": self.p,
                "seed": self._seed,
            },
        }


class SomeOf(Transform):
    """Pick ``n`` children at random (without replacement) and apply them in order.

    ``n`` may be an int or a ``(low, high)`` inclusive range — when a range,
    a value is sampled per call.
    """

    def __init__(
        self,
        transforms: Sequence[Transform],
        n: int | tuple[int, int] = 1,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.transforms: list[Transform] = list(transforms)
        if not self.transforms:
            raise ValueError("SomeOf requires at least one transform")
        for t in self.transforms:
            if not isinstance(t, Transform):
                raise TypeError(f"SomeOf expected Transform, got {type(t).__name__}")

        if isinstance(n, int):
            lo, hi = n, n
        else:
            lo, hi = int(n[0]), int(n[1])
        if not 0 <= lo <= hi <= len(self.transforms):
            raise ValueError(f"n={n} invalid for {len(self.transforms)} transforms")
        self.n_range: tuple[int, int] = (lo, hi)

        self._reseed_children()

    def _reseed_children(self) -> None:
        for t, child_rng in zip(self.transforms, derive_rng(self.rng, len(self.transforms))):
            t.set_rng(child_rng)

    def set_rng(self, rng: np.random.Generator) -> None:
        super().set_rng(rng)
        self._reseed_children()

    def apply(self, volume: MedVolume) -> MedVolume:
        lo, hi = self.n_range
        n = int(self.rng.integers(lo, hi + 1))
        if n == 0:
            return volume
        idxs = self.rng.choice(len(self.transforms), size=n, replace=False)
        idxs.sort()
        out = volume
        for i in idxs:
            out = self.transforms[int(i)].apply(out)
        return out

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.n_range
        n: Any = lo if lo == hi else list(self.n_range)
        return {
            "name": self.__class__.__name__,
            "params": {
                "transforms": [t.to_dict() for t in self.transforms],
                "n": n,
                "p": self.p,
                "seed": self._seed,
            },
        }


__all__ = ["Compose", "OneOf", "SomeOf"]
