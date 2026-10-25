"""Abstract base class that every MedAugmentX transform inherits from."""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from medaugmentx.core.tracing import ACTIVE_TRACE
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
        self._owner: weakref.ReferenceType[Transform] | None = None
        self.p: float = float(p)
        # Store the seed for serialisation (int or None only; Generator can't round-trip).
        self._seed: int | None = int(seed) if isinstance(seed, (int, np.integer)) else None
        self.rng: np.random.Generator = resolve_rng(seed)

    def __call__(self, volume: MedVolume) -> MedVolume:
        return self._run(volume)

    def _run(self, volume: MedVolume, *, force: bool = False) -> MedVolume:
        if not isinstance(volume, MedVolume):
            raise TypeError(f"Transform expects a MedVolume, got {type(volume).__name__}")
        trace = ACTIVE_TRACE.get()
        if not force and self.p < 1.0 and self.rng.random() >= self.p:
            if trace:
                trace(self, "skipped")
            return volume
        if trace:
            trace(self, "applied")
        out = self.apply(volume)
        if not isinstance(out, MedVolume):
            raise TypeError(f"{type(self).__name__}.apply must return a MedVolume")
        return out

    def _adopt_children(self, children: list[Transform]) -> None:
        seen = set()
        for child in children:
            if not isinstance(child, Transform):
                raise TypeError(f"{type(self).__name__} expected Transform instances")
            owner = getattr(child, "_owner", None)
            if id(child) in seen or (
                owner is not None and owner() is not None and owner() is not self
            ):
                raise ValueError(
                    "Transform instances cannot be shared between pipeline positions; construct or copy a new instance"
                )
            seen.add(id(child))
        for child in children:
            child._owner = weakref.ref(self)

    def __getstate__(self) -> dict[str, Any]:
        # Ownership is a construction guard, not serialized execution state.
        return {key: value for key, value in self.__dict__.items() if key != "_owner"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def reseed(self, seed: SeedLike) -> None:
        """Restart this pipeline's random sequence, recursively reseeding children."""
        self._seed = int(seed) if isinstance(seed, (int, np.integer)) else None
        self.set_rng(resolve_rng(seed))

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
            f"{k}={v!r}" for k, v in self.__dict__.items() if k != "rng" and not k.startswith("_")
        )
        return f"{self.__class__.__name__}({attrs})"

    def to_dict(self) -> dict[str, Any]:
        """Best-effort dictionary form of this transform's parameters.

        Subclasses whose stored attributes differ from constructor arguments
        must override this method to produce reconstructible parameters.
        """
        params = {k: v for k, v in self.__dict__.items() if k != "rng" and not k.startswith("_")}
        params["seed"] = self._seed
        return {"name": self.__class__.__name__, "params": params}
