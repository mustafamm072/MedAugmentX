"""Per-sample seeding and exact same-environment execution records."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import scipy

from medaugmentx._version import __version__
from medaugmentx.core import MedVolume, Transform
from medaugmentx.core.tracing import ACTIVE_TRACE


def environment_versions() -> dict[str, str]:
    """Versions needed to interpret an augmentation execution record."""
    return {
        "medaugmentx": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def seed_for_sample(
    seed: int, sample_id: str | int, *, epoch: int = 0, augmentation_index: int = 0
) -> int:
    """Stable seed independent of worker assignment, rank, and processing order.

    The same sample identifier/epoch/index deliberately yields the same draw on
    every worker. Use a distinct augmentation_index for independent replicas.
    Sample identifiers are hashed locally and are not written to a record.
    """
    for name, value in (
        ("seed", seed),
        ("epoch", epoch),
        ("augmentation_index", augmentation_index),
    ):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer")
    if not isinstance(sample_id, (str, int, np.integer)) or isinstance(sample_id, bool):
        raise TypeError("sample_id must be a stable string or integer")
    identity = str(sample_id) if isinstance(sample_id, str) else int(sample_id)
    data = json.dumps(
        [int(seed), identity, int(epoch), int(augmentation_index)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:16], "little")


def apply_for_sample(
    transform: Transform,
    volume: MedVolume,
    *,
    seed: int,
    sample_id: str | int,
    epoch: int = 0,
    augmentation_index: int = 0,
) -> MedVolume:
    """Apply an isolated policy copy with stable per-sample seeding.

    Does not advance the original transform. Copying large histogram references
    can be expensive; create worker-local policies for throughput-sensitive code.
    """
    isolated = copy.deepcopy(transform)
    isolated.reseed(
        seed_for_sample(seed, sample_id, epoch=epoch, augmentation_index=augmentation_index)
    )
    return isolated(volume.copy())


def _nodes(root: Transform, path: str = "root") -> Iterator[tuple[str, Transform]]:
    yield path, root
    if hasattr(root, "transforms"):
        for index, child in enumerate(root.transforms):
            yield from _nodes(child, f"{path}.{index}")
    elif hasattr(root, "transform"):
        yield from _nodes(root.transform, f"{path}.guarded")
    elif hasattr(root, "_inner"):
        yield from _nodes(root._inner, f"{path}.inner")


def _fingerprint(volume: MedVolume) -> str:
    from medaugmentx.serialization import _make_serialisable

    digest = hashlib.sha256()
    for name in (
        "image",
        "mask",
        "keypoints",
        "keypoint_labels",
        "bboxes",
        "bbox_labels",
        "affine",
    ):
        value = getattr(volume, name)
        digest.update(name.encode())
        if value is not None:
            arr = np.ascontiguousarray(value)
            digest.update(str((arr.shape, arr.dtype)).encode())
            # Object/string labels must not hash pointer addresses.
            digest.update(
                json.dumps(arr.tolist(), sort_keys=True).encode()
                if arr.dtype.kind == "O"
                else arr.tobytes()
            )
    metadata = _make_serialisable(volume.metadata)
    digest.update(
        json.dumps(
            [volume.spacing, metadata],
            sort_keys=True,
            default=str,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    )
    return digest.hexdigest()


def record_augmentation(
    transform: Transform, volume: MedVolume
) -> tuple[MedVolume, dict[str, Any]]:
    """Execute and record RNG checkpoints, branch decisions, policy, and hashes.

    Records do not embed image data or volume metadata. Policy parameters such
    as histogram references may contain data; inspect records before sharing.
    Checkpoints replay numerical random draws rather than transcribing every
    draw as a separate scalar parameter. Version/environment equality is required.
    """
    from medaugmentx.serialization import _make_serialisable, policy_document

    generators: dict[int, int] = {}
    states: list[dict[str, Any]] = []
    bindings: dict[str, int] = {}
    names = {}
    for path, node in _nodes(transform):
        names[id(node)] = path
        identity = id(node.rng)
        if identity not in generators:
            generators[identity] = len(states)
            states.append(
                {
                    "bit_generator": type(node.rng.bit_generator).__name__,
                    "state": _make_serialisable(copy.deepcopy(node.rng.bit_generator.state)),
                }
            )
        bindings[path] = generators[identity]
    trace: list[dict[str, str]] = []
    record = {
        "record_version": 1,
        "environment": environment_versions(),
        "policy": policy_document(transform),
        "rng_states": states,
        "rng_bindings": bindings,
        "input_sha256": _fingerprint(volume),
        "trace": trace,
    }
    token = ACTIVE_TRACE.set(
        lambda node, action: trace.append(
            {"path": names[id(node)], "name": type(node).__name__, "action": action}
        )
    )
    try:
        output = transform(volume.copy())
    finally:
        ACTIVE_TRACE.reset(token)
    record["output_sha256"] = _fingerprint(output)
    return output, record


def replay_augmentation(record: Mapping[str, Any], volume: MedVolume) -> MedVolume:
    """Replay a record, verifying environment, source input, and output fingerprints."""
    from medaugmentx.serialization import from_dict

    if record.get("record_version") != 1:
        raise ValueError("Unsupported augmentation record version")
    if record.get("environment") != environment_versions():
        raise ValueError("Replay requires the recorded package/Python/NumPy/SciPy environment")
    if record.get("input_sha256") != _fingerprint(volume):
        raise ValueError("Replay input fingerprint differs")
    transform = from_dict(record["policy"])
    generators = []
    for saved in record["rng_states"]:
        name = saved["bit_generator"]
        allowed = {"PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64"}
        if name not in allowed:
            raise ValueError("Unsupported recorded bit generator")
        generator = np.random.Generator(getattr(np.random, name)())
        generator.bit_generator.state = saved["state"]
        generators.append(generator)
    for path, node in _nodes(transform):
        index = record["rng_bindings"][path]
        if not isinstance(index, int) or not 0 <= index < len(generators):
            raise ValueError("Invalid RNG binding")
        node.rng = generators[index]
    output = transform(volume.copy())
    if _fingerprint(output) != record.get("output_sha256"):
        raise ValueError(
            "Replay output differs; custom transform state or environment may have changed"
        )
    return output
