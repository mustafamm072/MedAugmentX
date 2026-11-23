"""Pipeline builders: Compose, OneOf, SomeOf."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike, derive_rng
from medaugmentx.core.volume import FORBID_WARP, MedVolume

_SPATIAL_CHILD = (
    "PerChannel requires a transform that leaves shape, mask, geometry and targets "
    "unchanged; apply spatial transforms outside PerChannel"
)


class Compose(Transform):
    """Apply transforms sequentially.

    All children share a deterministic seeding chain derived from the
    top-level seed. Fresh equivalent pipelines reproduce the same sequence
    of draws in the same software environment; successive calls advance RNG
    state. Serialisation stores configuration, not the current RNG state.
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
                raise TypeError(f"Compose expected Transform instances, got {type(t).__name__}")
        self._adopt_children(self.transforms)
        self._reseed_children()

    def _reseed_children(self) -> None:
        if not self.transforms:
            return
        for t, child_rng in zip(self.transforms, derive_rng(self.rng, len(self.transforms))):
            t.set_rng(child_rng)

    def set_rng(self, rng: np.random.Generator) -> None:
        super().set_rng(rng)
        self._adopt_children(self.transforms)
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

        self._weights: list[float] | None = None
        if weights is None:
            self.weights = np.full(len(self.transforms), 1.0 / len(self.transforms))
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (len(self.transforms),):
                raise ValueError("weights length must match number of transforms")
            if not np.isfinite(w).all() or (w < 0).any() or not (w > 0).any():
                raise ValueError("weights must be finite, non-negative and sum to > 0")
            # Serialise the caller's weights: re-normalising already
            # normalised floats changes low bits and the policy hash.
            self._weights = w.tolist()
            w = w / w.max()
            self.weights = w / w.sum()

        self._adopt_children(self.transforms)
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
        return self.transforms[idx]._run(volume, force=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {
                "transforms": [t.to_dict() for t in self.transforms],
                "weights": self.weights.tolist() if self._weights is None else self._weights,
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

        from medaugmentx.core.parameters import validate_parameters

        validate_parameters(num_holes=n)
        if isinstance(n, (int, np.integer)):
            lo, hi = n, n
        else:
            lo, hi = int(n[0]), int(n[1])
        if not 0 <= lo <= hi <= len(self.transforms):
            raise ValueError(f"n={n} invalid for {len(self.transforms)} transforms")
        self.n_range: tuple[int, int] = (lo, hi)

        self._adopt_children(self.transforms)
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
            out = self.transforms[int(i)]._run(out, force=True)
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


class PerChannel(Transform):
    """Apply an intensity transform to each registered image channel.

    Spatial transforms already share one sampled geometry across the named
    :attr:`MedVolume.channels <medaugmentx.core.volume.MedVolume.channels>`,
    while intensity transforms act on the primary image only. Wrap an
    intensity or appearance policy in ``PerChannel`` to augment the primary
    image and selected channels inside one serialisable, replayable policy.

    By default every image receives its own draw from one deterministic
    stream, in order: the primary image, then the selected channels. With
    ``shared=True`` every image reuses the first image's random draws, which
    suits channels with a common intensity scale.

    The wrapped transform must leave the image shape, mask, geometry and
    targets unchanged; spatial transforms raise ``ValueError``. Metadata comes
    from the primary image's result, or is unchanged when it is excluded.

    Args:
        transform: Intensity transform or pipeline to apply per image.
        channels: Named channels to augment, in order. ``None`` selects every
            channel present in insertion order. Missing names raise ``KeyError``.
        include_primary: Also augment the primary image.
        shared: Reuse the same random draws for every selected image.
        p: Probability of applying the wrapper at all.
        seed: Seed for the wrapped transform's derived stream.
    """

    def __init__(
        self,
        transform: Transform,
        channels: Sequence[str] | None = None,
        include_primary: bool = True,
        shared: bool = False,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if not isinstance(transform, Transform):
            raise TypeError(f"PerChannel expects a Transform, got {type(transform).__name__}")
        if channels is not None:
            if isinstance(channels, str):
                raise TypeError("channels must be a sequence of names, not a string")
            names = tuple(channels)
            if any(not isinstance(name, str) or not name.strip() for name in names):
                raise ValueError("channel names must be non-empty strings")
            if len(set(names)) != len(names):
                raise ValueError("channel names must be distinct")
            channels = names
        self.channels: tuple[str, ...] | None = channels
        self.include_primary = bool(include_primary)
        self.shared = bool(shared)
        if not self.include_primary and self.channels == ():
            raise ValueError("PerChannel must select at least one image")
        self.transform = transform
        self._adopt_children([transform])
        self._reseed_child()

    def _reseed_child(self) -> None:
        (child_rng,) = derive_rng(self.rng, 1)
        self.transform.set_rng(child_rng)

    def set_rng(self, rng: np.random.Generator) -> None:
        super().set_rng(rng)
        self._reseed_child()

    def _generators(self) -> list[np.random.Generator]:
        found: dict[int, np.random.Generator] = {}
        pending: list[Transform] = [self.transform]
        while pending:
            node = pending.pop()
            found.setdefault(id(node.rng), node.rng)
            pending.extend(getattr(node, "transforms", ()))
            for attribute in ("transform", "_inner"):
                child = getattr(node, attribute, None)
                if isinstance(child, Transform):
                    pending.append(child)
        return list(found.values())

    def apply(self, volume: MedVolume) -> MedVolume:
        names = tuple(volume.channels) if self.channels is None else self.channels
        missing = [name for name in names if name not in volume.channels]
        if missing:
            raise KeyError(f"PerChannel channels not present in the volume: {missing}")
        targets: list[str | None] = ([None] if self.include_primary else []) + list(names)
        if not targets:
            return volume

        generators = self._generators() if self.shared else []
        saved = [copy.deepcopy(g.bit_generator.state) for g in generators]
        image, channels, metadata = volume.image, dict(volume.channels), volume.metadata
        for index, name in enumerate(targets):
            if index and self.shared:
                for generator, state in zip(generators, saved):
                    generator.bit_generator.state = copy.deepcopy(state)
            view = volume.select_channel(name)
            token = FORBID_WARP.set(_SPATIAL_CHILD)
            try:
                out = self.transform(view)
            finally:
                FORBID_WARP.reset(token)
            _check_intensity_only(view, out)
            if name is None:
                image, metadata = out.image, out.metadata
            else:
                channels[name] = out.image
        return volume.replace(image=image, channels=channels, metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {
                "transform": self.transform.to_dict(),
                "channels": None if self.channels is None else list(self.channels),
                "include_primary": self.include_primary,
                "shared": self.shared,
                "p": self.p,
                "seed": self._seed,
            },
        }


def _check_intensity_only(before: MedVolume, after: MedVolume) -> None:
    def same(a: np.ndarray | None, b: np.ndarray | None) -> bool:
        return a is b or (a is not None and b is not None and np.array_equal(a, b))

    if (
        after.shape != before.shape
        or after.channels
        or after.spacing != before.spacing
        or not all(
            same(getattr(before, field), getattr(after, field))
            for field in ("mask", "affine", "keypoints", "keypoint_labels", "bboxes", "bbox_labels")
        )
    ):
        raise ValueError(_SPATIAL_CHILD)


__all__ = ["Compose", "OneOf", "SomeOf", "PerChannel"]
