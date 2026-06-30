"""Helpers for inspecting transforms and nested augmentation pipelines."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from medaugmentx.core.base import Transform


@dataclass(frozen=True)
class PipelineStep:
    """One node in a transform or pipeline tree.

    Attributes:
        path: Tuple of child indices from the root. The root step uses ``()``;
            the first child of the root uses ``(0,)``; nested children use
            paths such as ``(1, 0)``.
        name: Transform class name from ``transform.to_dict()["name"]``.
        params: Parameters for this step, excluding nested ``transforms``.
        depth: Nesting depth, equal to ``len(path)``.
    """

    path: tuple[int, ...]
    name: str
    params: Mapping[str, Any]
    depth: int


def iter_pipeline(transform: Transform) -> Iterator[PipelineStep]:
    """Yield a depth-first view of a transform or nested pipeline.

    The iterator is backed by each transform's public ``to_dict()`` output, so
    it mirrors the same structure used for JSON/YAML serialisation.

    Args:
        transform: Any MedAugmentX ``Transform`` or pipeline container.

    Yields:
        ``PipelineStep`` objects in pre-order traversal.

    Raises:
        TypeError: If ``transform`` is not a ``Transform``.
        ValueError: If ``transform.to_dict()`` does not expose a transform name.
    """
    if not isinstance(transform, Transform):
        raise TypeError(
            f"iter_pipeline expects a Transform, got {type(transform).__name__}"
        )
    yield from _walk_dict(transform.to_dict(), ())


def pipeline_summary(transform: Transform, *, max_value_length: int = 72) -> str:
    """Return a compact multi-line summary of a transform or pipeline.

    Args:
        transform: Any MedAugmentX ``Transform`` or nested pipeline.
        max_value_length: Maximum rendered length for each parameter value.

    Returns:
        A stable, human-readable tree suitable for logs, experiment notes, and
        README snippets.
    """
    if max_value_length < 8:
        raise ValueError("max_value_length must be at least 8")

    lines = []
    for step in iter_pipeline(transform):
        indent = "  " * step.depth
        label = step.name if not step.path else f"{_format_path(step.path)} {step.name}"
        params = _format_params(step.params, max_value_length)
        lines.append(f"{indent}{label}{params}")
    return "\n".join(lines)


def _walk_dict(d: Mapping[str, Any], path: tuple[int, ...]) -> Iterator[PipelineStep]:
    name = d.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("pipeline dictionaries must contain a non-empty 'name'")

    raw_params = d.get("params", {})
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, Mapping):
        raise ValueError(f"{name} params must be a mapping")

    params = dict(raw_params)
    children = params.pop("transforms", None)
    yield PipelineStep(path=path, name=name, params=params, depth=len(path))

    if children is None:
        return
    if not isinstance(children, list):
        raise ValueError(f"{name} transforms must be a list")
    for index, child in enumerate(children):
        if not isinstance(child, Mapping):
            raise ValueError(f"{name} child {index} must be a mapping")
        yield from _walk_dict(child, (*path, index))


def _format_path(path: tuple[int, ...]) -> str:
    return ".".join(str(i) for i in path)


def _format_params(params: Mapping[str, Any], max_value_length: int) -> str:
    if not params:
        return ""
    rendered = [
        f"{key}={_truncate(_format_value(value), max_value_length)}"
        for key, value in params.items()
    ]
    return "(" + ", ".join(rendered) + ")"


def _format_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, np.generic):
        return repr(value.item())
    return repr(value)


def _truncate(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


__all__ = ["PipelineStep", "iter_pipeline", "pipeline_summary"]
