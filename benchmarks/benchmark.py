"""Speed benchmark for MedAugmentX transforms.

Times each registered leaf transform on a representative volume and prints a
table sorted by mean wall-clock time. Use this to spot regressions and to
check the Phase-3 acceptance target: every transform should complete in
< 500 ms on CPU for a 512x512x80 DBT volume.

Usage::

    python benchmarks/benchmark.py                  # default 64x256x256 volume
    python benchmarks/benchmark.py --shape 80 512 512   # acceptance-target size
    python benchmarks/benchmark.py --repeats 5 --no-3d-only-skip

The benchmark uses only NumPy/SciPy — no optional dependencies required.
"""
from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from medaugmentx import MedVolume
from medaugmentx.serialization import REGISTRY

# Constructor arguments for transforms that need them, keyed by class name.
# Anything not listed is instantiated with its defaults.
_CONSTRUCTOR_ARGS: dict[str, dict[str, Any]] = {
    "AnatomicCrop": {"size": (32, 128, 128)},
    "Resize": {"size": (40, 160, 160)},
    "Pad": {"size": (96, 320, 320)},
    "CenterCrop": {"size": (32, 128, 128)},
}

# Containers and transforms that need a reference / nested transforms are skipped.
_SKIP = {"Compose", "OneOf", "SomeOf", "HistogramMatch"}

# Transforms that require a 3D volume.
_REQUIRE_3D = {
    "SlabShift",
    "LimitedAngleBlur",
    "SliceDropout",
    "AnisotropicElastic",
    "CompressionVariation",
    "ReconStreak",
}


def _make_volume(shape: tuple[int, ...]) -> MedVolume:
    rng = np.random.default_rng(0)
    image = rng.random(shape, dtype=np.float64).astype(np.float32)
    mask = (image > 0.7).astype(np.uint8)
    spacing = (1.0, 0.1, 0.1) if len(shape) == 3 else (0.1, 0.1)
    return MedVolume(image=image, mask=mask, spacing=spacing, metadata={"modality": "DBT"})


def _time_transform(name: str, cls: type, volume: MedVolume, repeats: int) -> float | None:
    kwargs = dict(_CONSTRUCTOR_ARGS.get(name, {}))
    kwargs.setdefault("p", 1.0)
    kwargs.setdefault("seed", 0)
    try:
        transform = cls(**kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  ! could not construct {name}: {exc}")
        return None

    # Warm-up (JIT-free, but primes caches / first-call allocations).
    transform(volume)

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        transform(volume)
        best = min(best, time.perf_counter() - start)
    return best * 1000.0  # ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[64, 256, 256],
        help="Volume shape (2 or 3 ints). Default: 64 256 256.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timed runs per transform.")
    args = parser.parse_args()

    shape = tuple(args.shape)
    volume = _make_volume(shape)
    is_3d = len(shape) == 3

    print(f"MedAugmentX benchmark — volume shape {shape}, {args.repeats} repeats (best of)\n")
    results: list[tuple[str, float]] = []
    for name in sorted(REGISTRY):
        if name in _SKIP:
            continue
        if name in _REQUIRE_3D and not is_3d:
            continue
        cls = REGISTRY[name]
        ms = _time_transform(name, cls, volume, args.repeats)
        if ms is not None:
            results.append((name, ms))

    results.sort(key=lambda r: r[1], reverse=True)
    width = max(len(n) for n, _ in results)
    print(f"{'transform':<{width}}   time (ms)")
    print("-" * (width + 13))
    over_budget = []
    for name, ms in results:
        flag = "  <-- > 500 ms" if ms > 500 else ""
        if ms > 500:
            over_budget.append(name)
        print(f"{name:<{width}}   {ms:8.2f}{flag}")

    print()
    if over_budget:
        print(f"{len(over_budget)} transform(s) exceeded the 500 ms target: {', '.join(over_budget)}")
    else:
        print("All transforms completed within the 500 ms CPU target.")


if __name__ == "__main__":
    main()
