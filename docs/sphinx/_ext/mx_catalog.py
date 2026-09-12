"""Sphinx directives that build the transform reference from the package itself.

The transform list, the one-line summaries, the constructor parameters, and the
"does this move my annotations?" column are all read out of ``medaugmentx`` at
build time. Adding a transform to ``medaugmentx.transforms.__all__`` is enough to
make it appear in the catalog and in the generated API pages — there is no
hand-maintained list to keep in sync.

Two directives are provided:

``.. mx-transform-catalog::``
    A scannable table per category: transform, what it simulates, its knobs, and
    whether it warps geometric targets.

``.. mx-transform-api::``
    One ``autoclass`` block per transform, grouped under the same categories, so
    the full docstring and signature come straight from the source.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.util.nodes import nested_parse_with_titles

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

_GEOMETRY_MODULE = "medaugmentx.core.geometry"


def _transform_classes() -> list[type]:
    """Every publicly exported transform, in declaration order."""
    module = importlib.import_module("medaugmentx.transforms")
    return [getattr(module, name) for name in module.__all__]


def _category_of(cls: type) -> str:
    """Longest matching category prefix for a transform's defining module."""
    best = ""
    for key, _heading, _blurb in CATEGORIES:
        if cls.__module__.startswith(key) and len(key) > len(best):
            best = key
    return best


def _summary(cls: type) -> str:
    """First paragraph of the class docstring, as a single line."""
    doc = inspect.getdoc(cls) or ""
    paragraph = doc.split("\n\n", 1)[0]
    return " ".join(paragraph.split())


def _parameters(cls: type) -> list[str]:
    """Constructor parameters that actually configure the effect.

    ``p`` and ``seed`` are on every transform and carry no information here, so
    they are dropped to keep the column scannable.
    """
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return []
    return [name for name in signature.parameters if name not in {"p", "seed"}]


def _warps_targets(cls: type) -> bool:
    """Whether this transform maps keypoints and bounding boxes with the pixels.

    A transform can only move targets by building a point map from
    :mod:`medaugmentx.core.geometry` and handing it to :meth:`MedVolume.warp`,
    so the presence of that module in the defining module's namespace is an
    exact structural test — no source-text heuristics and nothing to keep in
    sync by hand. Both import styles are recognised: the module itself
    (``from medaugmentx.core import geometry``) and individual helpers pulled
    out of it.
    """
    module = inspect.getmodule(cls)
    if module is None:  # pragma: no cover - defensive
        return False
    for value in vars(module).values():
        if inspect.ismodule(value) and value.__name__ == _GEOMETRY_MODULE:
            return True
        if getattr(value, "__module__", None) == _GEOMETRY_MODULE:
            return True
    return False


def _escape(text: str) -> str:
    """Escape the one character that would break a list-table cell."""
    return text.replace("|", r"\|")


class _GeneratedDirective(Directive):
    """Directive whose body is generated reStructuredText."""

    has_content = False

    def lines(self) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def run(self) -> list[nodes.Node]:
        # The generated content depends on the installed package rather than on
        # any source file, so nothing extra needs to be registered as a
        # dependency: a changed package means a changed environment anyway.
        source = StringList(self.lines(), source="<mx_catalog>")
        container = nodes.section()
        container.document = self.state.document
        nested_parse_with_titles(self.state, source, container)
        return container.children


class TransformCatalog(_GeneratedDirective):
    """Render the scannable "which transform do I want?" tables."""

    def lines(self) -> list[str]:
        classes = _transform_classes()
        out: list[str] = []

        for key, heading, blurb in CATEGORIES:
            members = [c for c in classes if _category_of(c) == key]
            if not members:
                continue

            out += [heading, "-" * len(heading), "", blurb, ""]
            out += [
                ".. list-table::",
                "   :header-rows: 1",
                "   :widths: 22 44 24 10",
                "",
                "   * - Transform",
                "     - What it simulates",
                "     - Parameters",
                "     - Targets",
            ]
            for cls in members:
                params = _parameters(cls)
                rendered = ", ".join(f"``{p}``" for p in params) if params else "—"
                targets = "warped" if _warps_targets(cls) else "unchanged"
                out += [
                    f"   * - :class:`~medaugmentx.transforms.{cls.__name__}`",
                    f"     - {_escape(_summary(cls))}",
                    f"     - {rendered}",
                    f"     - {targets}",
                ]
            out.append("")

        return out


class TransformAPI(_GeneratedDirective):
    """Emit an ``autoclass`` block for every exported transform."""

    def lines(self) -> list[str]:
        classes = _transform_classes()
        out: list[str] = []

        for key, heading, blurb in CATEGORIES:
            members = [c for c in classes if _category_of(c) == key]
            if not members:
                continue

            out += [heading, "-" * len(heading), "", blurb, ""]
            for cls in members:
                # Only the class itself: ``apply`` is the internal hook that
                # subclasses override, carries no docstring of its own, and is
                # already explained once on the Transform base class. Callers
                # invoke the transform, not apply.
                out += [
                    f".. autoclass:: medaugmentx.transforms.{cls.__name__}",
                    "   :show-inheritance:",
                    "",
                ]

        return out


def setup(app: Any) -> dict[str, Any]:
    app.add_directive("mx-transform-catalog", TransformCatalog)
    app.add_directive("mx-transform-api", TransformAPI)
    return {
        "version": importlib.import_module("medaugmentx").__version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
