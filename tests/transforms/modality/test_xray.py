import numpy as np
import pytest

from medaugmentx.core import MedVolume
from medaugmentx.transforms import GridArtifact, ScatterSimulation


@pytest.fixture
def vol2d():
    img = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    mask = (img > 0.5).astype(np.uint8)
    return MedVolume(image=img, mask=mask)


@pytest.fixture
def vol3d():
    img = np.random.default_rng(1).random((6, 32, 32)).astype(np.float32)
    return MedVolume(image=img)


# ---- ScatterSimulation ----

def test_scatter_shape_dtype(vol2d):
    out = ScatterSimulation(sigma=(5.0, 10.0), seed=0)(vol2d)
    assert out.image.shape == vol2d.image.shape
    assert out.image.dtype == np.float32


def test_scatter_reduces_contrast(vol2d):
    # Scatter adds a smooth offset, raising the mean.
    out = ScatterSimulation(fraction=0.4, sigma=10.0, seed=0)(vol2d)
    assert out.image.mean() > vol2d.image.mean()


def test_scatter_mask_untouched(vol2d):
    out = ScatterSimulation(sigma=8.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.mask, vol2d.mask)


def test_scatter_fraction_zero_identity(vol2d):
    out = ScatterSimulation(fraction=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_scatter_deterministic(vol2d):
    a = ScatterSimulation(fraction=(0.1, 0.4), sigma=(5.0, 10.0), seed=2)(vol2d)
    b = ScatterSimulation(fraction=(0.1, 0.4), sigma=(5.0, 10.0), seed=2)(vol2d)
    np.testing.assert_array_equal(a.image, b.image)


def test_scatter_invalid_params_raise():
    with pytest.raises(ValueError):
        ScatterSimulation(sigma=0.0)
    with pytest.raises(ValueError):
        ScatterSimulation(fraction=-0.1)


def test_scatter_to_dict_round_trip():
    t = ScatterSimulation(fraction=(0.1, 0.4), sigma=(15.0, 40.0), p=0.5)
    t2 = ScatterSimulation(**t.to_dict()["params"])
    assert t2.fraction_range == t.fraction_range
    assert t2.sigma_range == t.sigma_range


# ---- GridArtifact ----

def test_grid_shape_dtype_3d(vol3d):
    out = GridArtifact(seed=0)(vol3d)
    assert out.image.shape == vol3d.image.shape
    assert out.image.dtype == np.float32


def test_grid_changes_image(vol2d):
    out = GridArtifact(amplitude=0.1, frequency=0.3, seed=0)(vol2d)
    assert not np.allclose(out.image, vol2d.image)


def test_grid_mask_untouched(vol2d):
    out = GridArtifact(seed=0)(vol2d)
    np.testing.assert_array_equal(out.mask, vol2d.mask)


def test_grid_amplitude_zero_identity(vol2d):
    out = GridArtifact(amplitude=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_grid_axis_orientation(vol2d):
    # "x" stripes vary along columns → rows are not identical; check it differs from "y".
    ox = GridArtifact(amplitude=0.1, frequency=0.3, axis="x", seed=0)(vol2d)
    oy = GridArtifact(amplitude=0.1, frequency=0.3, axis="y", seed=0)(vol2d)
    assert not np.allclose(ox.image, oy.image)


def test_grid_deterministic(vol2d):
    a = GridArtifact(amplitude=(0.03, 0.1), seed=5)(vol2d)
    b = GridArtifact(amplitude=(0.03, 0.1), seed=5)(vol2d)
    np.testing.assert_array_equal(a.image, b.image)


def test_grid_invalid_params_raise():
    with pytest.raises(ValueError):
        GridArtifact(frequency=0.0)
    with pytest.raises(ValueError):
        GridArtifact(axis="z")


def test_grid_to_dict_round_trip():
    t = GridArtifact(amplitude=(0.03, 0.1), frequency=(0.2, 0.45), axis="y", p=0.4)
    t2 = GridArtifact(**t.to_dict()["params"])
    assert t2.amplitude_range == t.amplitude_range
    assert t2.frequency_range == t.frequency_range
    assert t2.axis == t.axis
