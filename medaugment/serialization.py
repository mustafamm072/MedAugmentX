"""Pipeline serialisation — JSON round-trip and optional YAML support.

Usage::

    from medaugment import Compose
    from medaugment.transforms import GaussianNoise, GammaCorrection
    from medaugment.serialization import to_json, from_json

    pipeline = Compose([GaussianNoise(std=0.05), GammaCorrection()], seed=42)

    json_str = to_json(pipeline)
    pipeline2 = from_json(json_str)

Optional YAML support requires ``pip install pyyaml``.  All other functions
use only the Python standard library.

Registry
--------
:data:`REGISTRY` maps class names to classes.  Custom transforms must be
registered before deserialisation::

    from medaugment.serialization import REGISTRY
    REGISTRY["MyTransform"] = MyTransform
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medaugment.core.base import Transform

# ---------------------------------------------------------------------------
# Registry — populated below with all built-in transforms
# ---------------------------------------------------------------------------

REGISTRY: dict[str, type] = {}


def _register_builtins() -> None:
    # Core containers
    from medaugment.core.compose import Compose, OneOf, SomeOf

    REGISTRY.update({"Compose": Compose, "OneOf": OneOf, "SomeOf": SomeOf})

    # Spatial
    from medaugment.transforms.spatial.affine import RandomAffine
    from medaugment.transforms.spatial.crop import AnatomicCrop
    from medaugment.transforms.spatial.elastic import ElasticDeform
    from medaugment.transforms.spatial.flip import RandomFlip

    REGISTRY.update(
        {
            "RandomAffine": RandomAffine,
            "RandomFlip": RandomFlip,
            "AnatomicCrop": AnatomicCrop,
            "ElasticDeform": ElasticDeform,
        }
    )

    # Intensity
    from medaugment.transforms.intensity.bias_field import BiasField
    from medaugment.transforms.intensity.blur import GaussianBlur, SimulateLowResolution
    from medaugment.transforms.intensity.brightness_contrast import BrightnessContrast
    from medaugment.transforms.intensity.contrast import GammaCorrection
    from medaugment.transforms.intensity.noise import GaussianNoise, RicianNoise
    from medaugment.transforms.intensity.window_level import WindowLevel

    REGISTRY.update(
        {
            "GaussianNoise": GaussianNoise,
            "RicianNoise": RicianNoise,
            "GammaCorrection": GammaCorrection,
            "BiasField": BiasField,
            "WindowLevel": WindowLevel,
            "BrightnessContrast": BrightnessContrast,
            "GaussianBlur": GaussianBlur,
            "SimulateLowResolution": SimulateLowResolution,
        }
    )

    # Tomosynthesis
    from medaugment.transforms.modality.tomosynthesis.blur import LimitedAngleBlur
    from medaugment.transforms.modality.tomosynthesis.dropout import SliceDropout
    from medaugment.transforms.modality.tomosynthesis.elastic import AnisotropicElastic
    from medaugment.transforms.modality.tomosynthesis.slab import SlabShift

    REGISTRY.update(
        {
            "SlabShift": SlabShift,
            "LimitedAngleBlur": LimitedAngleBlur,
            "SliceDropout": SliceDropout,
            "AnisotropicElastic": AnisotropicElastic,
        }
    )

    # MRI
    from medaugment.transforms.modality.mri.ghosting import GhostingArtifact
    from medaugment.transforms.modality.mri.kspace import KSpaceDropout

    REGISTRY.update({"GhostingArtifact": GhostingArtifact, "KSpaceDropout": KSpaceDropout})

    # CT
    from medaugment.transforms.modality.ct.beam_hardening import BeamHardening

    REGISTRY.update({"BeamHardening": BeamHardening})


_register_builtins()


# ---------------------------------------------------------------------------
# from_dict — recursive reconstruction
# ---------------------------------------------------------------------------


def from_dict(d: dict[str, Any]) -> "Transform":
    """Reconstruct a :class:`~medaugment.core.base.Transform` from its dict form.

    The dict must have the structure produced by :meth:`Transform.to_dict`::

        {"name": "GaussianNoise", "params": {"std": 0.05, "p": 1.0}}

    Container transforms (``Compose``, ``OneOf``, ``SomeOf``) nest child
    dicts under ``params["transforms"]`` and are reconstructed recursively.

    Args:
        d: Dict as returned by ``transform.to_dict()``.

    Returns:
        A fully reconstructed :class:`~medaugment.core.base.Transform`.

    Raises:
        KeyError: If the transform name is not in :data:`REGISTRY`.
    """
    name: str = d["name"]
    if name not in REGISTRY:
        raise KeyError(
            f"Transform {name!r} is not in the serialisation registry.  "
            f"Register it via ``medaugment.serialization.REGISTRY[{name!r}] = MyClass``."
        )
    params: dict[str, Any] = dict(d.get("params", {}))

    # Containers carry nested child dicts — reconstruct them first.
    if name in ("Compose", "OneOf", "SomeOf"):
        child_dicts = params.pop("transforms", [])
        params["transforms"] = [from_dict(c) for c in child_dicts]

    cls = REGISTRY[name]
    return cls(**params)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy/tuple types to JSON-native equivalents."""
    import numpy as np

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


def to_json(transform: "Transform", indent: int = 2) -> str:
    """Serialise a transform (or pipeline) to a JSON string.

    Args:
        transform: Any :class:`~medaugment.core.base.Transform`, including
            :class:`~medaugment.core.compose.Compose` pipelines.
        indent: JSON indentation level.

    Returns:
        A JSON string that can be saved to disk and passed to
        :func:`from_json` to reconstruct the pipeline.
    """
    return json.dumps(_make_serialisable(transform.to_dict()), indent=indent)


def from_json(s: str) -> "Transform":
    """Reconstruct a transform from a JSON string produced by :func:`to_json`.

    Args:
        s: JSON string.

    Returns:
        The reconstructed :class:`~medaugment.core.base.Transform`.
    """
    return from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# YAML helpers (optional — requires PyYAML)
# ---------------------------------------------------------------------------


def to_yaml(transform: "Transform") -> str:
    """Serialise a transform to a YAML string.

    Requires ``pip install pyyaml``.

    Args:
        transform: Any :class:`~medaugment.core.base.Transform`.

    Returns:
        A YAML string.

    Raises:
        ImportError: If PyYAML is not installed.
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "YAML serialisation requires PyYAML: pip install pyyaml"
        ) from exc
    return yaml.dump(
        _make_serialisable(transform.to_dict()),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )


def from_yaml(s: str) -> "Transform":
    """Reconstruct a transform from a YAML string produced by :func:`to_yaml`.

    Requires ``pip install pyyaml``.

    Args:
        s: YAML string.

    Returns:
        The reconstructed :class:`~medaugment.core.base.Transform`.

    Raises:
        ImportError: If PyYAML is not installed.
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "YAML deserialisation requires PyYAML: pip install pyyaml"
        ) from exc
    return from_dict(yaml.safe_load(s))


__all__ = [
    "REGISTRY",
    "from_dict",
    "to_json",
    "from_json",
    "to_yaml",
    "from_yaml",
]
