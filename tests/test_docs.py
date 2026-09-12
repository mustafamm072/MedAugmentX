"""Guards on the inputs the generated documentation reads.

The transform catalog and the per-transform API pages are built from the
package itself (see ``docs/sphinx/_ext/mx_catalog.py``), so a transform cannot
be *missing* from the docs. It can, however, land there with an empty cell or in
no category at all, and that only shows up when someone looks at the rendered
site. These tests fail in CI instead.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

from medaugmentx import transforms as transforms_module

EXT_PATH = Path(__file__).resolve().parents[1] / "docs" / "sphinx" / "_ext" / "mx_catalog.py"


def _load_extension():
    spec = importlib.util.spec_from_file_location("mx_catalog", EXT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("mx_catalog", module)
    spec.loader.exec_module(module)
    return module


mx_catalog = _load_extension()

TRANSFORMS = [getattr(transforms_module, name) for name in transforms_module.__all__]
IDS = list(transforms_module.__all__)


@pytest.mark.parametrize("cls", TRANSFORMS, ids=IDS)
def test_transform_lands_in_a_catalog_category(cls):
    """A transform in an unrecognised subpackage would vanish from the catalog.

    The catalog groups by module prefix, so adding a new modality directory
    without extending ``CATEGORIES`` silently drops every transform in it.
    """
    assert mx_catalog._category_of(cls), (
        f"{cls.__name__} lives in {cls.__module__}, which matches no category in "
        f"mx_catalog.CATEGORIES — it would not appear in the transform catalog."
    )


@pytest.mark.parametrize("cls", TRANSFORMS, ids=IDS)
def test_transform_has_a_summary_line(cls):
    """The catalog's "What it simulates" column is the first docstring paragraph."""
    summary = mx_catalog._summary(cls)
    assert summary, f"{cls.__name__} has no class docstring to summarise."
    assert summary.endswith("."), (
        f"{cls.__name__} summary should be a sentence ending in a period: {summary!r}"
    )


@pytest.mark.parametrize("cls", TRANSFORMS, ids=IDS)
def test_transform_documents_every_parameter(cls):
    """Every constructor argument is described in the docstring.

    autodoc renders the Args section verbatim, so an undocumented parameter
    reaches the site as a bare name with no explanation.
    """
    doc = inspect.getdoc(cls) or ""
    undocumented = [
        name for name in mx_catalog._parameters(cls) if f"{name}:" not in doc
    ]
    assert not undocumented, (
        f"{cls.__name__} does not document {undocumented} in its Args section."
    )


def test_every_exported_transform_is_categorised_once():
    """Categories must not overlap, or a transform would be documented twice."""
    for cls in TRANSFORMS:
        matches = [
            key for key, _heading, _blurb in mx_catalog.CATEGORIES
            if cls.__module__.startswith(key)
        ]
        longest = max(matches, key=len)
        assert all(longest.startswith(m) for m in matches), (
            f"{cls.__name__} matches unrelated categories {matches}; "
            "the catalog would emit it more than once."
        )


def test_targets_column_matches_documented_behaviour():
    """The generated "Targets" column against the hand-written specification.

    ``docs/API_REFERENCE.md`` states which transforms warp keypoints and boxes.
    The catalog derives the same fact from whether the defining module uses
    ``medaugmentx.core.geometry``. If the two ever disagree, one of them is
    wrong and readers are being told two different things.
    """
    expected_warping = {
        "RandomFlip", "RandomAffine", "ElasticDeform", "AnatomicCrop",
        "CenterCrop", "Pad", "Resize", "SlabShift", "CompressionVariation",
    }
    actual = {cls.__name__ for cls in TRANSFORMS if mx_catalog._warps_targets(cls)}
    assert actual == expected_warping
