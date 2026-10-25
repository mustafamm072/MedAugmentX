"""Measure transform latency and memory in isolated processes; emit auditable JSON."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy

from medaugmentx import MedVolume, __version__
from medaugmentx.serialization import REGISTRY

_SKIP = {"Compose", "OneOf", "SomeOf", "Guard"}
_REQUIRE_3D = {
    "SlabShift",
    "LimitedAngleBlur",
    "SliceDropout",
    "AnisotropicElastic",
    "CompressionVariation",
    "ReconStreak",
}
_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def measure(name: str, shape: tuple[int, ...], repeats: int) -> dict:
    image = np.random.default_rng(0).random(shape).astype(np.float32)
    volume = MedVolume(
        image,
        mask=(image > 0.7).astype(np.uint8),
        spacing=(1.0, 0.5, 0.5) if len(shape) == 3 else (0.5, 0.5),
    )
    kwargs = {"p": 1.0, "seed": 0}
    factors = {"AnatomicCrop": 0.5, "CenterCrop": 0.5, "Resize": 0.625, "Pad": 1.5}
    if name in factors:
        kwargs["size"] = tuple(max(1, round(s * factors[name])) for s in shape)
    if name == "HistogramMatch":
        kwargs["reference_quantiles"] = np.linspace(0, 1, 256).tolist()
    transform = REGISTRY[name](**kwargs)
    transform(volume)  # warm-up excluded
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        transform(volume)
        timings.append((time.perf_counter() - start) * 1000)
    tracemalloc.start()
    transform(volume)
    _, tracked_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_bytes = None
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = int(rss if sys.platform == "darwin" else rss * 1024)
    except ImportError:
        pass
    return {
        "name": name,
        "policy": transform.to_dict(),
        "latency_ms": timings,
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "tracked_peak_bytes": tracked_peak,
        "process_peak_rss_bytes": rss_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs="+", type=int, default=[64, 256, 256])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if len(args.shape) not in (2, 3) or min(args.shape) < 1 or min(args.repeats, args.threads) < 1:
        parser.error("shape needs 2/3 positive axes; repeats and threads must be positive")
    if args.worker:
        print(json.dumps(measure(args.worker, tuple(args.shape), args.repeats)))
        return
    names = args.names or sorted(set(REGISTRY) - _SKIP)
    if set(names) - (set(REGISTRY) - _SKIP):
        parser.error("names must be registered leaf transforms")
    env = {**os.environ, **dict.fromkeys(_THREAD_KEYS, str(args.threads))}
    results = []
    for name in names:
        if len(args.shape) == 2 and name in _REQUIRE_3D:
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            name,
            "--repeats",
            str(args.repeats),
            "--shape",
            *map(str, args.shape),
        ]
        result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
        row = json.loads(result.stdout)
        results.append(row)
        print(f"{name:25s} median {row['median_ms']:9.2f} ms | p95 {row['p95_ms']:9.2f} ms")
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    report = {
        "schema_version": 1,
        "environment": {
            "medaugmentx": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "thread_environment": {k: env[k] for k in _THREAD_KEYS},
            "numpy_build_config": config.getvalue(),
        },
        "input": {
            "shape": args.shape,
            "dtype": "float32",
            "mask": "uint8",
            "targets": False,
            "seed": 0,
            "repeats": args.repeats,
            "warmups": 1,
        },
        "memory_scope": "Separate extra draw: tracemalloc peak; isolated process lifetime RSS includes interpreter, imports, input, warm-up and measurements. RSS unavailable on Windows.",
        "results": results,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
