"""Round-trip serialisation tests for all registered transforms."""
import json

import numpy as np
import pytest

from medaugment import Compose, MedVolume, OneOf, SomeOf
from medaugment.serialization import REGISTRY, from_dict, from_json, to_json
from medaugment.transforms import (
    AnatomicCrop,
    AnisotropicElastic,
    BeamHardening,
    BiasField,
    BrightnessContrast,
    ElasticDeform,
    GammaCorrection,
    GaussianBlur,
    GaussianNoise,
    GhostingArtifact,
    KSpaceDropout,
    LimitedAngleBlur,
    RandomAffine,
    RandomFlip,
    RicianNoise,
    SimulateLowResolution,
    SlabShift,
    SliceDropout,
    WindowLevel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def round_trip(transform):
    """Serialise to JSON and reconstruct."""
    return from_json(to_json(transform))


@pytest.fixture
def vol2d():
    img = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    return MedVolume(image=img)


@pytest.fixture
def vol3d():
    img = np.random.default_rng(0).random((8, 16, 16)).astype(np.float32)
    return MedVolume(image=img)


# ---------------------------------------------------------------------------
# Individual transform round-trips (to_dict → JSON → from_json)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transform", [
    GaussianNoise(std=0.05, p=0.9),
    GaussianNoise(std=(0.01, 0.05), relative=True, clip=(0.0, 1.0)),
    RicianNoise(std=0.02, p=0.8),
    RicianNoise(std=(0.005, 0.02), clip=(0.0, 1.0)),
    GammaCorrection(gamma=1.2, p=0.7),
    GammaCorrection(gamma=(0.8, 1.2), invert=True),
    RandomFlip(axes=("x",), p_per_axis=0.5),
    RandomFlip(axes=("x", "y"), p_per_axis=0.7, p=0.8),
    RandomAffine(rotation=10.0, p=0.7),
    RandomAffine(rotation=(-5.0, 5.0), scale=(0.9, 1.1), translation=(-0.05, 0.05)),
    AnatomicCrop(size=(16, 16), foreground_prob=0.5),
    ElasticDeform(alpha=30.0, sigma=4.0, p=0.6),
    ElasticDeform(alpha=(10.0, 10.0, 2.0), sigma=(4.0, 4.0, 1.0)),
    SlabShift(max_shift=3, p=0.5),
    SlabShift(max_shift=(0, 5)),
    LimitedAngleBlur(arc_degrees=(15.0, 25.0), p=0.7),
    LimitedAngleBlur(arc_degrees=20.0),
    SliceDropout(num_slices=2, p=0.4),
    SliceDropout(num_slices=(1, 3), affect_mask=True),
    AnisotropicElastic(alpha=(80.0, 80.0, 6.0), sigma=(8.0, 8.0, 2.0)),
    BiasField(alpha=0.3, coarse_shape=4, p=0.7),
    WindowLevel(center_shift_frac=0.1, width_scale=(0.8, 1.2), p=0.6),
    BrightnessContrast(brightness=0.05, contrast=(0.9, 1.1), p=0.5),
    GaussianBlur(sigma=(0.5, 1.5), p=0.6),
    SimulateLowResolution(zoom_range=(0.5, 0.9), per_axis=True, p=0.4),
    GhostingArtifact(ghost_intensity=(0.05, 0.15), ghost_shift=(8, 32), p=0.3),
    KSpaceDropout(dropout_fraction=(0.01, 0.05), phase_encode_axis="y", p=0.4),
    BeamHardening(alpha=(0.02, 0.08), power=2.0, p=0.3),
])
def test_leaf_transform_json_round_trip(transform):
    d = transform.to_dict()
    assert "name" in d
    assert "params" in d
    # Re-serialise through JSON (validates JSON-serializability)
    json_str = to_json(transform)
    loaded = json.loads(json_str)
    assert loaded["name"] == d["name"]
    # Reconstruct and check type
    reconstructed = from_json(json_str)
    assert type(reconstructed) is type(transform)


# ---------------------------------------------------------------------------
# Container (Compose / OneOf / SomeOf) round-trips
# ---------------------------------------------------------------------------


def test_compose_round_trip(vol2d):
    pipeline = Compose(
        [GaussianNoise(std=0.05), GammaCorrection(gamma=(0.9, 1.1))], seed=0
    )
    reconstructed = round_trip(pipeline)
    assert isinstance(reconstructed, Compose)
    assert len(reconstructed.transforms) == 2


def test_one_of_round_trip(vol2d):
    t = OneOf(
        [GaussianNoise(std=0.05), RicianNoise(std=0.02)],
        weights=[0.3, 0.7],
        p=0.8,
        seed=1,
    )
    rt = round_trip(t)
    assert isinstance(rt, OneOf)
    np.testing.assert_allclose(rt.weights, t.weights, atol=1e-6)


def test_some_of_round_trip(vol2d):
    t = SomeOf(
        [GaussianNoise(std=0.05), GammaCorrection(), RandomFlip()],
        n=(1, 2),
        p=0.9,
        seed=2,
    )
    rt = round_trip(t)
    assert isinstance(rt, SomeOf)
    assert rt.n_range == t.n_range


def test_nested_pipeline_round_trip(vol2d):
    pipeline = Compose(
        [
            RandomFlip(axes=("x",)),
            OneOf([GaussianNoise(std=0.02), RicianNoise(std=0.02)]),
            GammaCorrection(gamma=(0.9, 1.1)),
        ],
        seed=42,
    )
    rt = round_trip(pipeline)
    assert isinstance(rt, Compose)
    assert isinstance(rt.transforms[1], OneOf)


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


def test_registry_contains_all_known_transforms():
    expected = {
        "Compose", "OneOf", "SomeOf",
        "RandomAffine", "RandomFlip", "AnatomicCrop", "ElasticDeform",
        "GaussianNoise", "RicianNoise", "GammaCorrection",
        "BiasField", "WindowLevel", "BrightnessContrast",
        "GaussianBlur", "SimulateLowResolution",
        "SlabShift", "LimitedAngleBlur", "SliceDropout", "AnisotropicElastic",
        "GhostingArtifact", "KSpaceDropout", "BeamHardening",
    }
    assert expected.issubset(set(REGISTRY.keys()))


def test_unknown_transform_raises_key_error():
    with pytest.raises(KeyError, match="not in the serialisation registry"):
        from_dict({"name": "NonExistentTransform", "params": {}})


# ---------------------------------------------------------------------------
# JSON string validity
# ---------------------------------------------------------------------------


def test_to_json_produces_valid_json():
    t = Compose([GaussianNoise(std=0.05), GammaCorrection()], seed=0)
    s = to_json(t)
    parsed = json.loads(s)  # must not raise
    assert parsed["name"] == "Compose"


def test_from_json_reconstructs_compose_pipeline():
    t = Compose([BiasField(alpha=0.3), WindowLevel(p=0.6)], seed=5)
    s = to_json(t)
    rt = from_json(s)
    assert isinstance(rt, Compose)
    assert len(rt.transforms) == 2
