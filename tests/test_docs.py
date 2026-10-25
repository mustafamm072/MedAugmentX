"""Guards on the inputs the generated documentation reads.

The transform catalog and the per-transform API pages are built from the
package itself (see ``docs/sphinx/_ext/``), so a transform cannot be *missing*
from the docs. It can, however, land there with an empty cell or in no category
at all, and that only shows up when someone looks at the rendered site. These
tests fail in CI instead.

They import ``mx_inventory``, which depends on nothing but the standard library
and ``medaugmentx``, so they run on every supported Python with only the
``test`` extra installed — Sphinx is not needed and must not be imported here.
The docs tree is absent when the suite is run against an installed wheel
outside the source checkout; there is no documentation to guard in that case,
so the module skips.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

from medaugmentx import transforms as transforms_module

EXT_DIR = Path(__file__).resolve().parents[1] / "docs" / "sphinx" / "_ext"
INVENTORY_PATH = EXT_DIR / "mx_inventory.py"

if not INVENTORY_PATH.is_file():
    pytest.skip(
        "docs/ is not part of the distribution; nothing to guard when the test "
        "suite runs against an installed wheel",
        allow_module_level=True,
    )


def _load_inventory():
    spec = importlib.util.spec_from_file_location("mx_inventory", INVENTORY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inventory = _load_inventory()

TRANSFORMS = [getattr(transforms_module, name) for name in transforms_module.__all__]
IDS = list(transforms_module.__all__)


@pytest.mark.parametrize("cls", TRANSFORMS, ids=IDS)
def test_transform_lands_in_a_catalog_category(cls):
    """A transform in an unrecognised subpackage would vanish from the catalog.

    The catalog groups by module prefix, so adding a new modality directory
    without extending ``CATEGORIES`` silently drops every transform in it.
    """
    assert inventory.category_of(cls), (
        f"{cls.__name__} lives in {cls.__module__}, which matches no category in "
        f"mx_inventory.CATEGORIES — it would not appear in the transform catalog."
    )


@pytest.mark.parametrize("cls", TRANSFORMS, ids=IDS)
def test_transform_has_a_summary_line(cls):
    """The catalog's "What it simulates" column is the first docstring paragraph."""
    summary = inventory.summary(cls)
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
    undocumented = [name for name in inventory.parameters(cls) if f"{name}:" not in doc]
    assert not undocumented, f"{cls.__name__} does not document {undocumented} in its Args section."


def test_categories_do_not_overlap():
    """Overlapping categories would emit a transform twice."""
    for cls in TRANSFORMS:
        matches = [
            key for key, _heading, _blurb in inventory.CATEGORIES if cls.__module__.startswith(key)
        ]
        longest = max(matches, key=len)
        assert all(longest.startswith(m) for m in matches), (
            f"{cls.__name__} matches unrelated categories {matches}; "
            "the catalog would emit it more than once."
        )


def test_targets_column_matches_documented_behaviour():
    """The generated "Targets" column against the hand-written specification.

    ``docs/API_REFERENCE.md`` states which transforms warp keypoints and boxes.
    The catalog uses geometry imports and recognises delegating wrappers.
    If the two ever disagree, one of them is
    wrong and readers are being told two different things.
    """
    expected_warping = {
        "RandomFlip",
        "RandomAffine",
        "ElasticDeform",
        "AnisotropicElastic",
        "AnatomicCrop",
        "CenterCrop",
        "Pad",
        "Resize",
        "SlabShift",
        "CompressionVariation",
    }
    actual = {cls.__name__ for cls in TRANSFORMS if inventory.warps_targets(cls)}
    assert actual == expected_warping


def test_extension_keeps_sphinx_out_of_the_inventory():
    """mx_inventory must stay importable without Sphinx installed.

    This suite runs on every matrix Python with only the ``test`` extra, where
    Sphinx is absent. If a docutils or sphinx import creeps into the inventory,
    collection breaks on all of them at once.
    """
    source = INVENTORY_PATH.read_text(encoding="utf-8")
    for forbidden in ("import docutils", "import sphinx", "from docutils", "from sphinx"):
        assert forbidden not in source, (
            f"mx_inventory.py must not contain {forbidden!r}; Sphinx-dependent "
            "code belongs in mx_catalog.py."
        )
