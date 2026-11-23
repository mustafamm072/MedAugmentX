"""PerChannel applies intensity policies to registered channels in one policy."""

import numpy as np
import pytest

from medaugmentx import (
    Compose,
    Guard,
    MedVolume,
    PerChannel,
    apply_for_sample,
    iter_pipeline,
    record_augmentation,
    replay_augmentation,
)
from medaugmentx.serialization import from_dict, from_json, to_json
from medaugmentx.transforms import (
    GammaCorrection,
    GaussianNoise,
    NormalizeIntensity,
    RandomAffine,
    RandomFlip,
)


@pytest.fixture
def volume():
    rng = np.random.default_rng(3)
    image = rng.random((4, 6, 7)).astype(np.float32) + 0.5
    mask = (image > 1.0).astype(np.uint8)
    return MedVolume(
        image,
        mask,
        channels={"flair": image.copy(), "t2": image.copy()},
        keypoints=np.array([[1.0, 2.0, 3.0]]),
        affine=np.diag([2.0, 1.0, 1.0, 1.0]),
    )


def test_independent_draws_per_image_leave_annotations(volume):
    out = PerChannel(GaussianNoise(std=0.1), seed=4)(volume)
    assert not np.array_equal(out.image, volume.image)
    assert not np.array_equal(out.channels["flair"], volume.channels["flair"])
    assert not np.array_equal(out.image, out.channels["flair"])
    assert not np.array_equal(out.channels["flair"], out.channels["t2"])
    assert out.mask is volume.mask
    np.testing.assert_array_equal(out.keypoints, volume.keypoints)
    np.testing.assert_array_equal(out.affine, volume.affine)
    assert list(out.channels) == ["flair", "t2"]


def test_shared_draws_match_across_identical_images(volume):
    pipeline = PerChannel(
        Compose([GaussianNoise(std=0.1, p=0.5), GammaCorrection(gamma=(0.5, 2.0))]),
        shared=True,
        seed=9,
    )
    for _ in range(4):
        out = pipeline(volume)
        np.testing.assert_array_equal(out.image, out.channels["flair"])
        np.testing.assert_array_equal(out.image, out.channels["t2"])


def test_selection_order_and_primary_exclusion(volume):
    out = PerChannel(GaussianNoise(std=0.1), channels=["t2"], include_primary=False, seed=1)(volume)
    np.testing.assert_array_equal(out.image, volume.image)
    np.testing.assert_array_equal(out.channels["flair"], volume.channels["flair"])
    assert not np.array_equal(out.channels["t2"], volume.channels["t2"])
    excluded = PerChannel(NormalizeIntensity(), include_primary=False, seed=0)(volume)
    assert "intensity_domain" not in excluded.metadata


def test_primary_metadata_comes_from_primary_result(volume):
    out = PerChannel(NormalizeIntensity(), seed=0)(volume)
    assert out.metadata["intensity_domain"] == "normalized"
    for array in (out.image, *out.channels.values()):
        assert 0.0 <= array.min() and array.max() <= 1.0


def test_volume_without_channels_matches_wrapped_transform():
    image = np.random.default_rng(0).random((5, 6)).astype(np.float32)
    plain = MedVolume(image)
    wrapped = PerChannel(GaussianNoise(std=0.2), seed=11)
    from medaugmentx.core.utils import derive_rng

    reference = GaussianNoise(std=0.2)
    (child,) = derive_rng(np.random.default_rng(11), 1)
    reference.set_rng(child)
    np.testing.assert_array_equal(wrapped(plain).image, reference(plain).image)


def test_rejects_spatial_children_and_bad_configuration(volume):
    with pytest.raises(ValueError, match="spatial transforms outside PerChannel"):
        PerChannel(RandomFlip(p_per_axis=1.0), seed=0)(volume)
    with pytest.raises(ValueError, match="spatial transforms outside PerChannel"):
        PerChannel(RandomAffine(rotation=10), seed=0)(volume)
    with pytest.raises(KeyError, match="missing"):
        PerChannel(GaussianNoise(), channels=["missing"])(volume)
    with pytest.raises(TypeError):
        PerChannel(GaussianNoise(), channels="flair")
    with pytest.raises(TypeError):
        PerChannel("noise")
    with pytest.raises(ValueError, match="distinct"):
        PerChannel(GaussianNoise(), channels=["a", "a"])
    with pytest.raises(ValueError, match="at least one"):
        PerChannel(GaussianNoise(), channels=[], include_primary=False)


def test_serialization_inspection_and_seed_round_trip(volume):
    pipeline = Compose(
        [
            RandomFlip(p_per_axis=0.5),
            PerChannel(GaussianNoise(std=0.05), channels=["flair"], shared=True),
        ],
        seed=21,
    )
    restored = from_json(to_json(pipeline))
    assert restored.to_dict() == pipeline.to_dict()
    assert from_dict(pipeline.to_dict()).to_dict() == pipeline.to_dict()
    names = [step.name for step in iter_pipeline(pipeline)]
    assert names == ["Compose", "RandomFlip", "PerChannel", "GaussianNoise"]
    first = apply_for_sample(pipeline, volume, seed=5, sample_id="case")
    again = apply_for_sample(restored, volume, seed=5, sample_id="case")
    np.testing.assert_array_equal(first.image, again.image)
    np.testing.assert_array_equal(first.channels["flair"], again.channels["flair"])


def test_record_and_replay_with_shared_state_restoration(volume):
    pipeline = Guard(PerChannel(GaussianNoise(std=0.05), shared=True), seed=2)
    output, record = record_augmentation(pipeline, volume)
    replayed = replay_augmentation(record, volume)
    np.testing.assert_array_equal(output.channels["t2"], replayed.channels["t2"])
    np.testing.assert_array_equal(output.image, replayed.image)


@pytest.mark.parametrize(
    "make",
    [lambda: RandomAffine(rotation=45.0), lambda: RandomFlip(axes=("x",), p_per_axis=1.0)],
    ids=["affine", "flip"],
)
def test_rejects_shape_preserving_spatial_child_without_annotations(make):
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    volume = MedVolume(image.copy(), affine=np.eye(4), channels={"t2": image.copy()})
    with pytest.raises(ValueError, match="spatial transforms outside PerChannel"):
        PerChannel(make(), seed=0)(volume)
    with pytest.raises(ValueError, match="spatial transforms outside PerChannel"):
        PerChannel(Guard(Compose([GaussianNoise(), make()])), seed=0)(volume)
    # The restriction is scoped to the wrapped call only.
    out = make()(volume)
    np.testing.assert_array_equal(out.image, out.channels["t2"])
