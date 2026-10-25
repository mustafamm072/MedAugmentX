import json
from pathlib import Path

import numpy as np
import pytest

from medaugmentx.core import MedVolume
from medaugmentx.phantoms import ct_phantom, dbt_phantom, mri_phantom, xray_phantom

PHANTOMS = [mri_phantom, ct_phantom, xray_phantom, dbt_phantom]
NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("factory", PHANTOMS, ids=lambda f: f.__name__)
def test_phantom_is_a_finite_volume(factory):
    volume = factory()
    assert isinstance(volume, MedVolume)
    assert volume.image.dtype == np.float32
    assert np.isfinite(volume.image).all()
    assert volume.image.std() > 0, "a constant phantom would teach nothing"
    assert len(volume.spacing) == volume.ndim


@pytest.mark.parametrize("factory", PHANTOMS, ids=lambda f: f.__name__)
def test_phantom_is_deterministic(factory):
    # Reproducibility is the whole point: docs and tutorials must not drift
    # between runs or machines.
    np.testing.assert_array_equal(factory().image, factory().image)


@pytest.mark.parametrize("factory", PHANTOMS, ids=lambda f: f.__name__)
def test_phantom_declares_its_modality(factory):
    assert factory().modality in {"MR", "CT", "DX", "DBT"}


@pytest.mark.parametrize(
    "factory", [mri_phantom, ct_phantom, xray_phantom], ids=lambda f: f.__name__
)
def test_two_dimensional_phantoms_honour_size(factory):
    assert factory().shape == (256, 256)
    assert factory(size=64).shape == (64, 64)


def test_ct_phantom_spans_a_plausible_hounsfield_range():
    image = ct_phantom().image
    assert image.min() < -900, "should contain air near -1000 HU"
    assert image.max() > 500, "should contain a dense structure"
    assert -1024 <= image.min() and image.max() <= 3071


def test_dbt_phantom_is_anisotropic_3d():
    volume = dbt_phantom()
    assert volume.is_3d and volume.shape == (64, 128, 256)
    # The DBT transforms rely on through-plane spacing exceeding in-plane.
    assert volume.spacing[0] > volume.spacing[-1]


def test_dbt_phantom_seed_changes_only_lesions():
    a, b = dbt_phantom(), dbt_phantom(seed=7)
    assert not np.array_equal(a.image, b.image)
    assert a.shape == b.shape


@pytest.mark.skipif(not NOTEBOOKS, reason="notebooks/ not present in this tree")
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_are_committed_without_outputs(path):
    # Outputs are stripped on purpose (clean diffs, small repo); CI executes the
    # notebooks instead. This catches an accidental commit from a live session.
    notebook = json.loads(path.read_text())
    for number, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] == "code":
            assert not cell.get("outputs"), f"cell {number} has committed outputs"
            assert cell.get("execution_count") is None, f"cell {number} has an execution count"
