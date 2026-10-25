"""Pipeline serialisation — JSON round-trip and optional YAML support.

Usage::

    from medaugmentx import Compose
    from medaugmentx.transforms import GaussianNoise, GammaCorrection
    from medaugmentx.serialization import to_json, from_json

    pipeline = Compose([GaussianNoise(std=0.05), GammaCorrection()], seed=42)

    json_str = to_json(pipeline)
    pipeline2 = from_json(json_str)

Optional YAML support requires ``pip install pyyaml``.  All other functions
use only the Python standard library.

Registry
--------
:data:`REGISTRY` maps class names to classes.  Custom transforms must be
registered before deserialisation.  The recommended way is the
:func:`register_transform` decorator, which validates the class and guards
against silently clobbering a built-in::

    from medaugmentx.core import Transform
    from medaugmentx.serialization import register_transform

    @register_transform
    class MyTransform(Transform):
        ...

Direct assignment (``REGISTRY["MyTransform"] = MyTransform``) still works for
backward compatibility.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload

if TYPE_CHECKING:
    from medaugmentx.core.base import Transform

    T = TypeVar("T", bound=type[Transform])

# ---------------------------------------------------------------------------
# Registry — populated below with all built-in transforms
# ---------------------------------------------------------------------------

REGISTRY: dict[str, type] = {}
SCHEMA_VERSION = 1


def _register_builtins() -> None:
    # Core containers
    from medaugmentx.core.compose import Compose, OneOf, SomeOf

    REGISTRY.update({"Compose": Compose, "OneOf": OneOf, "SomeOf": SomeOf})

    # Spatial
    from medaugmentx.transforms.spatial.affine import RandomAffine
    from medaugmentx.transforms.spatial.crop import AnatomicCrop
    from medaugmentx.transforms.spatial.dropout import CoarseDropout
    from medaugmentx.transforms.spatial.elastic import ElasticDeform
    from medaugmentx.transforms.spatial.flip import RandomFlip
    from medaugmentx.transforms.spatial.resize import CenterCrop, Pad, Resize

    REGISTRY.update(
        {
            "RandomAffine": RandomAffine,
            "RandomFlip": RandomFlip,
            "AnatomicCrop": AnatomicCrop,
            "ElasticDeform": ElasticDeform,
            "CoarseDropout": CoarseDropout,
            "Resize": Resize,
            "Pad": Pad,
            "CenterCrop": CenterCrop,
        }
    )

    # Intensity
    from medaugmentx.transforms.intensity.bias_field import BiasField
    from medaugmentx.transforms.intensity.blur import (
        GaussianBlur,
        MedianBlur,
        SimulateLowResolution,
    )
    from medaugmentx.transforms.intensity.brightness_contrast import BrightnessContrast
    from medaugmentx.transforms.intensity.clahe import CLAHEContrast
    from medaugmentx.transforms.intensity.contrast import GammaCorrection
    from medaugmentx.transforms.intensity.histogram import HistogramMatch
    from medaugmentx.transforms.intensity.noise import GaussianNoise, RicianNoise
    from medaugmentx.transforms.intensity.sharpen import Sharpen
    from medaugmentx.transforms.intensity.window_level import WindowLevel

    REGISTRY.update(
        {
            "GaussianNoise": GaussianNoise,
            "RicianNoise": RicianNoise,
            "GammaCorrection": GammaCorrection,
            "BiasField": BiasField,
            "WindowLevel": WindowLevel,
            "BrightnessContrast": BrightnessContrast,
            "GaussianBlur": GaussianBlur,
            "MedianBlur": MedianBlur,
            "SimulateLowResolution": SimulateLowResolution,
            "Sharpen": Sharpen,
            "CLAHEContrast": CLAHEContrast,
            "HistogramMatch": HistogramMatch,
        }
    )

    from medaugmentx.transforms.intensity.normalize import NormalizeIntensity

    REGISTRY["NormalizeIntensity"] = NormalizeIntensity

    # Tomosynthesis
    from medaugmentx.transforms.modality.tomosynthesis.blur import LimitedAngleBlur
    from medaugmentx.transforms.modality.tomosynthesis.compression import CompressionVariation
    from medaugmentx.transforms.modality.tomosynthesis.dropout import SliceDropout
    from medaugmentx.transforms.modality.tomosynthesis.elastic import AnisotropicElastic
    from medaugmentx.transforms.modality.tomosynthesis.recon_streak import ReconStreak
    from medaugmentx.transforms.modality.tomosynthesis.slab import SlabShift

    REGISTRY.update(
        {
            "SlabShift": SlabShift,
            "LimitedAngleBlur": LimitedAngleBlur,
            "SliceDropout": SliceDropout,
            "AnisotropicElastic": AnisotropicElastic,
            "CompressionVariation": CompressionVariation,
            "ReconStreak": ReconStreak,
        }
    )

    # MRI
    from medaugmentx.transforms.modality.mri.ghosting import GhostingArtifact
    from medaugmentx.transforms.modality.mri.kspace import KSpaceDropout
    from medaugmentx.transforms.modality.mri.motion import MRIMotion

    REGISTRY.update(
        {
            "GhostingArtifact": GhostingArtifact,
            "KSpaceDropout": KSpaceDropout,
            "MRIMotion": MRIMotion,
        }
    )

    # CT
    from medaugmentx.transforms.modality.ct.beam_hardening import BeamHardening
    from medaugmentx.transforms.modality.ct.metal import MetalStreak

    REGISTRY.update({"BeamHardening": BeamHardening, "MetalStreak": MetalStreak})

    # X-ray (DXR)
    from medaugmentx.transforms.modality.xray.grid import GridArtifact
    from medaugmentx.transforms.modality.xray.scatter import ScatterSimulation

    REGISTRY.update({"ScatterSimulation": ScatterSimulation, "GridArtifact": GridArtifact})

    # Safety guard (wraps a single transform + a validator)
    from medaugmentx.validation import Guard

    REGISTRY.update({"Guard": Guard})


_register_builtins()


# ---------------------------------------------------------------------------
# register_transform — decorator for custom transforms
# ---------------------------------------------------------------------------


@overload
def register_transform(cls: T) -> T: ...
@overload
def register_transform(
    cls: None = ..., *, name: str | None = ..., override: bool = ...
) -> Callable[[T], T]: ...


def register_transform(
    cls: type | None = None,
    *,
    name: str | None = None,
    override: bool = False,
) -> Any:
    """Register a custom :class:`~medaugmentx.core.base.Transform` for serialisation.

    Usable bare or with arguments::

        @register_transform
        class MyTransform(Transform): ...

        @register_transform(name="LegacyName")
        class MyTransform(Transform): ...

    The class is added to :data:`REGISTRY` under ``name`` (defaulting to the
    class's own ``__name__``) and returned unchanged, so it can be used as a
    decorator. To round-trip cleanly the class must implement ``to_dict`` and
    accept its ``to_dict`` ``params`` back through ``__init__``.

    Args:
        cls: The transform class (supplied automatically when used bare).
        name: Registry key. Defaults to ``cls.__name__``. Must match the
            ``"name"`` field the class emits from ``to_dict`` for round-trips
            to work.
        override: Allow replacing an existing registry entry. Defaults to
            ``False`` so a name collision with a built-in (or another custom
            transform) raises instead of silently shadowing it.

    Returns:
        The class itself (bare form) or a decorator (parametrised form).

    Raises:
        TypeError: If ``cls`` is not a subclass of ``Transform``.
        KeyError: If ``name`` is already registered to a *different* class and
            ``override`` is ``False``.
    """
    # Import here to avoid a circular import at module load time.
    from medaugmentx.core.base import Transform

    def _decorate(klass: type) -> type:
        if not (isinstance(klass, type) and issubclass(klass, Transform)):
            raise TypeError(f"register_transform expects a Transform subclass, got {klass!r}")
        key = name if name is not None else klass.__name__
        existing = REGISTRY.get(key)
        if existing is not None and existing is not klass and not override:
            raise KeyError(
                f"Transform name {key!r} is already registered to "
                f"{existing.__name__!r}. Pass override=True to replace it, or "
                f"choose a different name."
            )
        REGISTRY[key] = klass
        return klass

    # Bare usage: @register_transform
    if cls is not None:
        return _decorate(cls)
    # Parametrised usage: @register_transform(name=..., override=...)
    return _decorate


# ---------------------------------------------------------------------------
# from_dict — recursive reconstruction
# ---------------------------------------------------------------------------


def from_dict(d: dict[str, Any]) -> Transform:
    """Reconstruct a :class:`~medaugmentx.core.base.Transform` from its dict form.

    The dict must have the structure produced by :meth:`Transform.to_dict`::

        {"name": "GaussianNoise", "params": {"std": 0.05, "p": 1.0}}

    Container transforms (``Compose``, ``OneOf``, ``SomeOf``) nest child
    dicts under ``params["transforms"]`` and are reconstructed recursively.

    Args:
        d: Dict as returned by ``transform.to_dict()``.

    Returns:
        A fully reconstructed :class:`~medaugmentx.core.base.Transform`.

    Raises:
        KeyError: If the transform name is not in :data:`REGISTRY`.
    """
    from medaugmentx.core.base import Transform

    _validate_values(d)
    if isinstance(d, dict) and "schema_version" in d:
        from medaugmentx._version import __version__

        if set(d) != {"schema_version", "library_version", "pipeline", "policy_sha256"}:
            raise ValueError("Malformed versioned policy fields")
        if type(d["schema_version"]) is not int or d["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported policy schema version")
        version = d["library_version"]
        if not isinstance(version, str) or version.split(".")[0] != __version__.split(".")[0]:
            raise ValueError("Policy library major version is incompatible")
        if d["policy_sha256"] != _policy_hash(d["pipeline"]):
            raise ValueError("Policy hash mismatch")
        if version != __version__:
            warnings.warn(
                "Policy was created with a different package version; seeded output may change",
                UserWarning,
                stacklevel=2,
            )
        d = d["pipeline"]
    nodes = 0

    def build(node: Any, depth: int = 0) -> Transform:
        nonlocal nodes
        nodes += 1
        if depth > 64 or nodes > 10000:
            raise ValueError("Pipeline exceeds the limit of 64 nesting levels or 10000 transforms")
        if not isinstance(node, dict):
            raise ValueError("Each transform must be an object with name and params fields")
        if set(node) - {"name", "params"}:
            raise ValueError("Unknown transform fields; expected only name and params")
        name = node.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Transform name must be a non-empty string")
        if name not in REGISTRY:
            raise KeyError(f"Transform {name!r} is not in the serialisation registry.")
        raw_params = node.get("params", {})
        if not isinstance(raw_params, dict) or not all(isinstance(k, str) for k in raw_params):
            raise ValueError("Transform params must be an object with string keys")
        params = dict(raw_params)
        if name in ("Compose", "OneOf", "SomeOf"):
            children = params.get("transforms", [])
            if not isinstance(children, (list, tuple)):
                raise ValueError("Container transforms must be a list")
            params["transforms"] = [build(child, depth + 1) for child in children]
        if name == "Guard" and "transform" in params:
            params["transform"] = build(params["transform"], depth + 1)
        cls = REGISTRY[name]
        if not isinstance(cls, type) or not issubclass(cls, Transform):
            raise TypeError(f"Registry entry {name!r} must be a Transform subclass")
        return cls(**params)

    return build(d)


def _validate_values(value: Any) -> None:
    """Bound traversal and reject nonfinite values/cycles before constructors run."""
    active: set[int] = set()
    count = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal count
        count += 1
        if depth > 256 or count > 1_000_000:
            raise ValueError("Policy parameter structure is too large or deeply nested")
        if isinstance(item, Real) and not isinstance(item, int) and not math.isfinite(float(item)):
            raise ValueError("Policy numbers must be finite")
        if isinstance(item, (dict, list, tuple)):
            identity = id(item)
            if identity in active:
                raise ValueError("Policy nesting cannot contain cycles")
            active.add(identity)
            for child in item.values() if isinstance(item, dict) else item:
                visit(child, depth + 1)
            active.remove(identity)
        elif hasattr(item, "dtype"):
            import numpy as np

            if (
                isinstance(item, np.ndarray)
                and item.dtype.kind in "fc"
                and not np.isfinite(item).all()
            ):
                raise ValueError("Policy numbers must be finite")

    visit(value, 0)


def _policy_hash(node: Any) -> str:
    canonical = json.dumps(
        _make_serialisable(node), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def policy_document(transform: Transform) -> dict[str, Any]:
    """Versioned policy envelope with package version and a canonical SHA-256 hash."""
    from medaugmentx._version import __version__

    node = _make_serialisable(transform.to_dict())
    _validate_values(node)
    return {
        "schema_version": SCHEMA_VERSION,
        "library_version": __version__,
        "pipeline": node,
        "policy_sha256": _policy_hash(node),
    }


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy/tuple types to JSON-native equivalents."""
    import numpy as np

    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, list):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    return obj


def _check_document_size(s: str) -> None:
    if not isinstance(s, str):
        raise TypeError("Pipeline document must be a string")
    if len(s) > 1_000_000:
        raise ValueError("Pipeline document exceeds 1000000 characters")


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON constant {value!r} is not supported")


def to_json(transform: Transform, indent: int = 2) -> str:
    """Serialise a transform (or pipeline) to a JSON string.

    Args:
        transform: Any :class:`~medaugmentx.core.base.Transform`, including
            :class:`~medaugmentx.core.compose.Compose` pipelines.
        indent: JSON indentation level.

    Returns:
        A JSON string that can be saved to disk and passed to
        :func:`from_json` to reconstruct the pipeline.
    """
    return json.dumps(policy_document(transform), indent=indent, allow_nan=False)


def from_json(s: str) -> Transform:
    """Reconstruct a transform from a JSON string produced by :func:`to_json`.

    Args:
        s: JSON string.

    Returns:
        The reconstructed :class:`~medaugmentx.core.base.Transform`.
    """
    _check_document_size(s)
    try:
        parsed = json.loads(s, parse_constant=_reject_json_constant)
    except RecursionError as exc:
        raise ValueError("Pipeline document is nested too deeply") from exc
    if isinstance(parsed, dict) and "schema_version" not in parsed:
        warnings.warn(
            "Loading an unversioned pre-1.0 policy; v1 geometry and preset semantics apply. "
            "Re-export and review the migrated policy before use.",
            UserWarning,
            stacklevel=2,
        )
    return from_dict(parsed)


# ---------------------------------------------------------------------------
# YAML helpers (optional — requires PyYAML)
# ---------------------------------------------------------------------------


def to_yaml(transform: Transform) -> str:
    """Serialise a transform to a YAML string.

    Requires ``pip install pyyaml``.

    Args:
        transform: Any :class:`~medaugmentx.core.base.Transform`.

    Returns:
        A YAML string.

    Raises:
        ImportError: If PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML serialisation requires PyYAML: pip install pyyaml") from exc
    return yaml.safe_dump(
        policy_document(transform),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )


def from_yaml(s: str) -> Transform:
    """Reconstruct a transform from a YAML string produced by :func:`to_yaml`.

    Requires ``pip install pyyaml``.

    Args:
        s: YAML string.

    Returns:
        The reconstructed :class:`~medaugmentx.core.base.Transform`.

    Raises:
        ImportError: If PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML deserialisation requires PyYAML: pip install pyyaml") from exc
    _check_document_size(s)
    try:
        parsed = yaml.safe_load(s)
    except RecursionError as exc:
        raise ValueError("Pipeline document is nested too deeply") from exc
    if isinstance(parsed, dict) and "schema_version" not in parsed:
        warnings.warn(
            "Loading an unversioned pre-1.0 policy; review v1 migration semantics",
            UserWarning,
            stacklevel=2,
        )
    return from_dict(parsed)


__all__ = [
    "REGISTRY",
    "SCHEMA_VERSION",
    "policy_document",
    "register_transform",
    "from_dict",
    "to_json",
    "from_json",
    "to_yaml",
    "from_yaml",
]
