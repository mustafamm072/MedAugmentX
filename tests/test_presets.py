"""Smoke tests for pre-built modality pipelines."""
import numpy as np
import pytest

from medaugment import Compose, MedVolume
from medaugment.presets import ct_pipeline, dbt_pipeline, dxr_pipeline, mri_pipeline
from medaugment.serialization import from_json, to_json


@pytest.fixture
def vol2d():
    rng = np.random.default_rng(0)
    img = rng.random((64, 64)).astype(np.float32)
    mask = (img > 0.5).astype(np.uint8)
    return MedVolume(image=img, mask=mask, spacing=(0.7, 0.7), metadata={"modality": "DX"})


@pytest.fixture
def vol3d():
    rng = np.random.default_rng(1)
    img = rng.random((16, 32, 32)).astype(np.float32)
    mask = (img > 0.7).astype(np.uint8)
    return MedVolume(image=img, mask=mask, spacing=(2.5, 0.7, 0.7), metadata={"modality": "MR"})


@pytest.fixture
def dbt_vol():
    rng = np.random.default_rng(2)
    img = rng.random((20, 64, 64)).astype(np.float32)
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[5:15, 20:40, 20:40] = 1
    return MedVolume(
        image=img, mask=mask, spacing=(1.0, 0.1, 0.1), metadata={"modality": "DBT"}
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_mri_pipeline_returns_compose():
    assert isinstance(mri_pipeline(seed=0), Compose)


def test_ct_pipeline_returns_compose():
    assert isinstance(ct_pipeline(seed=0), Compose)


def test_dxr_pipeline_returns_compose():
    assert isinstance(dxr_pipeline(seed=0), Compose)


def test_dbt_pipeline_returns_compose():
    assert isinstance(dbt_pipeline(seed=0), Compose)


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


def test_mri_pipeline_runs_on_3d_volume(vol3d):
    out = mri_pipeline(seed=0)(vol3d)
    assert out.image.shape == vol3d.image.shape
    assert out.image.dtype == np.float32


def test_ct_pipeline_runs_on_3d_volume(vol3d):
    out = ct_pipeline(seed=0)(vol3d)
    assert out.image.shape == vol3d.image.shape


def test_dxr_pipeline_runs_on_2d_volume(vol2d):
    out = dxr_pipeline(seed=0)(vol2d)
    assert out.image.shape == vol2d.image.shape
    assert out.image.dtype == np.float32


def test_dbt_pipeline_runs_on_dbt_volume(dbt_vol):
    out = dbt_pipeline(seed=0)(dbt_vol)
    assert out.image.shape == dbt_vol.image.shape


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_mri_pipeline_deterministic(vol3d):
    a = mri_pipeline(seed=42)(vol3d)
    b = mri_pipeline(seed=42)(vol3d)
    np.testing.assert_array_equal(a.image, b.image)


def test_ct_pipeline_deterministic(vol3d):
    a = ct_pipeline(seed=7)(vol3d)
    b = ct_pipeline(seed=7)(vol3d)
    np.testing.assert_array_equal(a.image, b.image)


def test_dxr_pipeline_deterministic(vol2d):
    a = dxr_pipeline(seed=3)(vol2d)
    b = dxr_pipeline(seed=3)(vol2d)
    np.testing.assert_array_equal(a.image, b.image)


def test_dbt_pipeline_deterministic(dbt_vol):
    a = dbt_pipeline(seed=99)(dbt_vol)
    b = dbt_pipeline(seed=99)(dbt_vol)
    np.testing.assert_array_equal(a.image, b.image)


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


def test_mri_pipeline_serialisable():
    pipeline = mri_pipeline(seed=0)
    rt = from_json(to_json(pipeline))
    assert isinstance(rt, Compose)
    assert len(rt.transforms) == len(pipeline.transforms)


def test_ct_pipeline_serialisable():
    pipeline = ct_pipeline(seed=0)
    rt = from_json(to_json(pipeline))
    assert isinstance(rt, Compose)


def test_dxr_pipeline_serialisable():
    pipeline = dxr_pipeline(seed=0)
    rt = from_json(to_json(pipeline))
    assert isinstance(rt, Compose)


def test_dbt_pipeline_serialisable():
    pipeline = dbt_pipeline(seed=0)
    rt = from_json(to_json(pipeline))
    assert isinstance(rt, Compose)
