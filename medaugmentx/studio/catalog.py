"""Transform metadata for the studio editor, read from the package itself.

Everything here is derived from :data:`medaugmentx.serialization.REGISTRY`,
constructor signatures, ``Args:`` docstring sections and the packaged model
cards, so a transform added to the registry appears in the studio without any
hand-maintained list.
"""

from __future__ import annotations

import inspect
import math
import re
from typing import Any

from medaugmentx.cards import list_model_cards, model_card
from medaugmentx.core.base import Transform
from medaugmentx.core.parameters import NDIMAGE_MODES
from medaugmentx.serialization import REGISTRY, WRAPPER_NAMES

# Module prefix -> display category. The longest matching prefix wins.
CATEGORIES: tuple[tuple[str, str], ...] = (
    ("medaugmentx.core", "Pipeline"),
    ("medaugmentx.validation", "Pipeline"),
    ("medaugmentx.transforms.spatial", "Spatial"),
    ("medaugmentx.transforms.intensity", "Intensity"),
    ("medaugmentx.transforms.modality.mri", "MRI"),
    ("medaugmentx.transforms.modality.ct", "CT"),
    ("medaugmentx.transforms.modality.xray", "X-ray (DXR)"),
    ("medaugmentx.transforms.modality.tomosynthesis", "Tomosynthesis (DBT)"),
)
CATEGORY_ORDER = tuple(dict.fromkeys(label for _, label in CATEGORIES)) + ("Custom",)

CONTAINER_NAMES = ("Compose", "OneOf", "SomeOf")
NESTING = {name: "transforms" for name in CONTAINER_NAMES}
NESTING.update({name: "transform" for name in WRAPPER_NAMES})

_PAD_MODES = (
    "constant",
    "edge",
    "reflect",
    "symmetric",
    "wrap",
    "linear_ramp",
    "maximum",
    "mean",
    "median",
    "minimum",
)
# Allowed values for string parameters, verified against constructor checks.
_CHOICES: dict[str, tuple[str, ...]] = {
    "mode": tuple(sorted(NDIMAGE_MODES)),
    "space": ("physical", "voxel"),
    "translation_units": ("fraction", "voxels", "mm"),
    "label_axis": ("x", "y", "z"),
    "units": ("voxels", "mm"),
    "bbox_mode": ("conservative", "corners", "reject"),
    "axis": ("x", "y"),
    "phase_encode_axis": ("x", "y"),
    "on_fail": ("raise", "warn", "revert", "retry"),
}
_CHOICE_OVERRIDES: dict[tuple[str, str], tuple[str, ...]] = {("Pad", "mode"): _PAD_MODES}
_ROLE = re.compile(r":[a-z]+:`(~?)([^`<]+?)(?:\s*<[^>]*>)?`")
_ARG_LINE = re.compile(r"^(\*{0,2}\w+)(?:\s*\([^)]*\))?:\s*(.*)$")


def category_of(cls: type) -> str:
    """Display category for a registered transform class."""
    best, label = "", "Custom"
    for prefix, name in CATEGORIES:
        if cls.__module__.startswith(prefix) and len(prefix) > len(best):
            best, label = prefix, name
    return label


def plain_text(text: str) -> str:
    """Drop reStructuredText roles and literal markers for display."""
    # Like Sphinx: a leading "~" shows only the last dotted component.
    text = _ROLE.sub(lambda m: m.group(2).split(".")[-1] if m.group(1) else m.group(2), text)
    return text.replace("``", "").replace("**", "")


def summary_of(cls: type) -> str:
    """First docstring paragraph as one line of plain text."""
    doc = inspect.getdoc(cls) or ""
    return plain_text(" ".join(doc.split("\n\n", 1)[0].split()))


def parse_args_section(doc: str) -> dict[str, str]:
    """Map parameter names to their Google-style ``Args:`` descriptions."""
    descriptions: dict[str, str] = {}
    lines = doc.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Args:")
    except StopIteration:
        return descriptions
    base: int | None = None
    current: str | None = None
    for line in lines[start + 1 :]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if base is None:
            base = indent
        if indent < base:
            break
        match = _ARG_LINE.match(line.strip()) if indent == base else None
        if match:
            current = match.group(1).lstrip("*")
            descriptions[current] = match.group(2).strip()
        elif current is not None:
            descriptions[current] = f"{descriptions[current]} {line.strip()}".strip()
    return descriptions


def _kind(default: Any, annotation: str) -> str:
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, str):
        return "str"
    if default is inspect.Parameter.empty and annotation in ("bool", "int", "float", "str"):
        return annotation
    return "json"


def _json_default(value: Any) -> Any:
    from medaugmentx.serialization import _make_serialisable

    if value is inspect.Parameter.empty:
        return None
    value = _make_serialisable(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value if isinstance(value, (bool, int, float, str, list, dict, type(None))) else None


def describe_transform(name: str) -> dict[str, Any]:
    """Editor metadata for one registered transform."""
    cls = REGISTRY[name]
    doc = inspect.getdoc(cls) or ""
    descriptions = parse_args_section(doc)
    params = []
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        signature = None
    for parameter in [] if signature is None else signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.name == NESTING.get(name):
            continue
        annotation = parameter.annotation
        annotation = (
            annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
        )
        entry: dict[str, Any] = {
            "name": parameter.name,
            "kind": _kind(parameter.default, annotation),
            "default": _json_default(parameter.default),
            "required": parameter.default is inspect.Parameter.empty,
            "annotation": annotation,
            "description": plain_text(descriptions.get(parameter.name, "")),
        }
        choices = _CHOICE_OVERRIDES.get((name, parameter.name)) or _CHOICES.get(parameter.name)
        if choices and entry["kind"] == "str":
            entry["choices"] = list(choices)
        params.append(entry)
    card = model_card(name) if name in list_model_cards() else None
    return {
        "name": name,
        "category": category_of(cls),
        "summary": summary_of(cls),
        "container": NESTING.get(name),
        "params": params,
        "card": card,
    }


def catalog() -> list[dict[str, Any]]:
    """Metadata for every registered transform, grouped in category order."""
    entries = []
    for name, cls in REGISTRY.items():
        if isinstance(cls, type) and issubclass(cls, Transform):
            entries.append(describe_transform(name))
    rank = {label: index for index, label in enumerate(CATEGORY_ORDER)}
    return sorted(entries, key=lambda entry: (rank.get(entry["category"], len(rank)),))


def children_of(node: dict[str, Any]) -> list[tuple[str, int | None, dict[str, Any]]]:
    """Nested child nodes as ``(param, index, node)``; index is None for wrappers."""
    params = node.get("params") or {}
    key = NESTING.get(node.get("name", ""))
    if key == "transforms":
        return [
            ("transforms", index, child)
            for index, child in enumerate(params.get("transforms") or [])
            if isinstance(child, dict)
        ]
    if key == "transform" and isinstance(params.get("transform"), dict):
        return [("transform", None, params["transform"])]
    return []


__all__ = [
    "CATEGORIES",
    "CATEGORY_ORDER",
    "CONTAINER_NAMES",
    "NESTING",
    "catalog",
    "category_of",
    "children_of",
    "describe_transform",
    "parse_args_section",
    "plain_text",
    "summary_of",
]
