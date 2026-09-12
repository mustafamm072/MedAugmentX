"""Sphinx directives that build the transform reference from the package itself.

The facts come from :mod:`mx_inventory` — the transform list, the one-line
summaries, the constructor parameters, and the "does this move my annotations?"
column are all read out of ``medaugmentx`` at build time. Adding a transform to
``medaugmentx.transforms.__all__`` is enough to make it appear in the catalog
and in the generated API pages; there is no hand-maintained list to keep in
sync.

Two directives are provided:

``.. mx-transform-catalog::``
    A scannable table per category: transform, what it simulates, its knobs, and
    whether it warps geometric targets.

``.. mx-transform-api::``
    One ``autoclass`` block per transform, grouped under the same categories, so
    the full docstring and signature come straight from the source.

Sphinx and docutils are imported here and nowhere else in ``_ext``, so the test
suite can check the same facts without them installed.
"""
from __future__ import annotations

import importlib
from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.util.nodes import nested_parse_with_titles

import mx_inventory


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
        classes = mx_inventory.transform_classes()
        out: list[str] = []

        for key, heading, blurb in mx_inventory.CATEGORIES:
            members = mx_inventory.members_of(key, classes)
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
                params = mx_inventory.parameters(cls)
                rendered = ", ".join(f"``{p}``" for p in params) if params else "—"
                targets = "warped" if mx_inventory.warps_targets(cls) else "unchanged"
                out += [
                    f"   * - :class:`~medaugmentx.transforms.{cls.__name__}`",
                    f"     - {_escape(mx_inventory.summary(cls))}",
                    f"     - {rendered}",
                    f"     - {targets}",
                ]
            out.append("")

        return out


class TransformAPI(_GeneratedDirective):
    """Emit an ``autoclass`` block for every exported transform."""

    def lines(self) -> list[str]:
        classes = mx_inventory.transform_classes()
        out: list[str] = []

        for key, heading, blurb in mx_inventory.CATEGORIES:
            members = mx_inventory.members_of(key, classes)
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
