"""Abstract base class that every MedAugment transform inherits from."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from medaugmentx.core.utils import SeedLike, resolve_rng
from medaugmentx.core.volume import MedVolume


class Transform(ABC):
    """Base class for all augmentations.

    Subclasses must override :meth:`apply` and accept ``p`` and ``seed``
    through ``super().__init__``. The base class handles probabilistic
    application, seeding, and serialisation.

    Probabilistic application is gated on ``self.rng`` so that two transforms
    in the same :class:`Compose` with the same seed do not share a random
    stream — see :func:`medaugmentx.core.utils.derive_rng`.

    Example:

        class MyShift(Transform):
            def __init__(self, max_shift=0.1, p=1.0, seed=None):
                super().__init__(p=p, seed=seed)
                self.max_shift = max_shift

            def apply(self, volume):
                delta = self.rng.uniform(-self.max_shift, self.max_shift)
                return volume.replace(image=volume.image + delta)
    """

    def __init__(self, p: float = 1.0, seed: SeedLike = None) -> None:
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p: float = float(p)
        # Store the seed for serialisation (int or None only; Generator can't round-trip).
        self._seed: int | None = seed if isinstance(seed, (int, type(None))) else None
        self.rng: np.random.Generator = resolve_rng(seed)

    def __call__(self, volume: MedVolume) -> MedVolume:
        if not isinstance(volume, MedVolume):
            raise TypeError(f"Transform expects a MedVolume, got {type(volume).__name__}")
        if self.p < 1.0 and self.rng.random() >= self.p:
            return volume
        return self.apply(volume)

    @abstractmethod
    def apply(self, volume: MedVolume) -> MedVolume:
        """Perform the transform unconditionally — already past probability gate."""

    def set_rng(self, rng: np.random.Generator) -> None:
        """Reseed this transform with a specific :class:`numpy.random.Generator`.

        Used by :class:`Compose` to give each child its own deterministic stream.
        """
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        self.rng = rng

    def __repr__(self) -> str:
        attrs = ", ".join(
            f"{k}={v!r}"
            for k, v in self.__dict__.items()
            if k != "rng" and not k.startswith("_")
        )
        return f"{self.__class__.__name__}({attrs})"

    def to_dict(self) -> dict[str, Any]:
        """Best-effort dictionary form of this transform's parameters.

        Phase 1 ships only this introspection helper; full YAML serialisation
        and round-tripping arrive in Phase 2.
        """
        params = {
            k: v for k, v in self.__dict__.items() if k != "rng" and not k.startswith("_")
        }
        return {"name": self.__class__.__name__, "params": params}
