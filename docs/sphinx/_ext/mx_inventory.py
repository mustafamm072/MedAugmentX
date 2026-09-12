"""Facts about the shipped transforms, read out of the package.

This module deliberately imports **nothing but the standard library and
medaugmentx**. The Sphinx directives in :mod:`mx_catalog` build the docs from
it, and ``tests/test_docs.py`` checks the same facts on every supported Python
version — where only the ``test`` extra is installed and Sphinx is absent. Any
docutils or sphinx import belongs in ``mx_catalog``, not here.
"""
from __future__ import annotations

import importlib
import inspect

# Category key -> (heading, blurb). The key is matched against the *defining*
# module of each transform, which is what determines the category; the import
# path users are told to use stays the flat ``medaugmentx.transforms`` namespace.
CATEGORIES: list[tuple[str, str, str]] = [
    (
        "medaugmentx.transforms.spatial",
        "Spatial",
        "Geometry: resampling, cropping, flipping, and deformation. Masks are "
        "resampled with nearest-neighbour interpolation so label values are "
        "never blended.",
    ),
    (
        "medaugmentx.transforms.intensity",
        "Intensity",
        "Signal-domain changes that leave geometry alone — noise, contrast, "
        "blur, and bias. Masks and geometric targets always pass through "
        "untouched.",
    ),
    (
        "medaugmentx.transforms.modality.mri",
        "MRI",
        "Acquisition artifacts specific to magnetic resonance: k-space "
        "corruption, ghosting along the phase-encode axis, and rigid motion.",
    ),
    (
        "medaugmentx.transforms.modality.ct",
        "CT",
        "Polychromatic-beam and high-density-implant effects seen on computed "
        "tomography reconstructions.",
    ),
    (
        "medaugmentx.transforms.modality.xray",
        "X-ray (DXR)",
        "Projection-radiography effects: Compton scatter and anti-scatter grid "
        "line artifacts.",
    ),
    (
        "medaugmentx.transforms.modality.tomosynthesis",
        "Tomosynthesis (DBT)",
        "Limited-angle reconstruction effects. These act on the reconstructed "
        "volume and respect its strongly anisotropic voxel spacing.",
    ),
]

GEOMETRY_MODULE = "medaugmentx.core.geometry"


def transform_classes() -> list[type]:
    """Every publicly exported transform, in declaration order."""
    module = importlib.import_module("medaugmentx.transforms")
    return [getattr(module, name) for name in module.__all__]


def category_of(cls: type) -> str:
    """Longest matching category prefix for a transform's defining module."""
    best = ""
    for key, _heading, _blurb in CATEGORIES:
        if cls.__module__.startswith(key) and len(key) > len(best):
            best = key
    return best


def members_of(category: str, classes: list[type]) -> list[type]:
    """Transforms belonging to one category, in declaration order."""
    return [cls for cls in classes if category_of(cls) == category]


def summary(cls: type) -> str:
    """First paragraph of the class docstring, as a single line."""
    doc = inspect.getdoc(cls) or ""
    paragraph = doc.split("\n\n", 1)[0]
    return " ".join(paragraph.split())


def parameters(cls: type) -> list[str]:
    """Constructor parameters that actually configure the effect.

    ``p`` and ``seed`` are on every transform and carry no information here, so
    they are dropped to keep the catalog column scannable.
    """
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return []
    return [name for name in signature.parameters if name not in {"p", "seed"}]


def warps_targets(cls: type) -> bool:
    """Whether this transform maps keypoints and bounding boxes with the pixels.

    A transform can only move targets by building a point map from
    :mod:`medaugmentx.core.geometry` and handing it to ``MedVolume.warp``, so
    the presence of that module in the defining module's namespace is an exact
    structural test — no source-text heuristics and nothing to keep in sync by
    hand. Both import styles are recognised: the module itself
    (``from medaugmentx.core import geometry``) and individual helpers pulled
    out of it.
    """
    module = inspect.getmodule(cls)
    if module is None:  # pragma: no cover - defensive
        return False
    for value in vars(module).values():
        if inspect.ismodule(value) and value.__name__ == GEOMETRY_MODULE:
            return True
        if getattr(value, "__module__", None) == GEOMETRY_MODULE:
            return True
    return False
