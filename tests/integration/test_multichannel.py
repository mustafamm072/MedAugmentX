"""Registered channels use the same spatial draws as the single-image API."""

import numpy as np
import pytest

from medaugmentx import (
    Compose,
    MedVolume,
    VolumeValidator,
    apply_for_sample,
    record_augmentation,
    replay_augmentation,
    volume_summary,
)
from medaugmentx.transforms import (
    AnatomicCrop,
    AnisotropicElastic,
    CenterCrop,
    ClassBalancedCrop,
    CoarseDropout,
    CompressionVariation,
    ElasticDeform,
    GaussianNoise,
    Pad,
    RandomAffine,
    RandomFlip,
    ResampleToGrid,
    ResampleToSpacing,
    Resize,
    SlabShift,
    SliceDropout,
)


@pytest.fixture
def registered():
    shape = (4, 9, 12)
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    other = 10 + np.sqrt(image)
    mask = np.zeros(shape, dtype=np.uint64)
    mask[1, 4, 6] = 2**64 - 2
    affine = np.array([[0.1, 0, 0.5, 10], [0, 0.75, 0.1, 0], [2, 0.1, 0, 5], [0, 0, 0, 1]])
    return MedVolume(
        image,
        mask,
        affine=affine,
        channels={"flair": other},
        keypoints=np.array([[1, 4, 6]]),
        bboxes=np.array([[0, 2, 3, 3, 7, 10]]),
    )


@pytest.mark.parametrize(
    "transform",
    [
        RandomFlip(p_per_axis=1),
        RandomAffine(rotation=12, translation=0.1),
        ElasticDeform(alpha=0.15, sigma=1),
        AnisotropicElastic(alpha=(0.1, 0.2, 0.2), sigma=(1, 1, 1)),
        AnatomicCrop((3, 5, 7), foreground_prob=1),
        ClassBalancedCrop((6, 6, 7), foreground_prob=1),
        CenterCrop((3, 6, 8)),
        Pad((6, 13, 14), mode="reflect"),
        Resize((3, 7, 9), clip_output=True),
        ResampleToSpacing((1.5, 0.9, 0.8), clip_output=True),
        ResampleToGrid((6, 10, 7), np.eye(4)),
        CoarseDropout(num_holes=3, hole_size=(0.25, 0.55), fill_mask=True),
        CompressionVariation(scale=(0.7, 1.3)),
        SlabShift(max_shift=(1, 1)),
        SliceDropout(2, affect_mask=True),
    ],
    ids=lambda transform: type(transform).__name__,
)
def test_registered_channel_matches_one_identical_spatial_draw(registered, transform):
    draw = dict(seed=219, sample_id="paired", epoch=2)
    joint = apply_for_sample(transform, registered, **draw)
    primary = apply_for_sample(transform, registered.select_channel(), **draw)
    flair = apply_for_sample(transform, registered.select_channel("flair"), **draw)

    np.testing.assert_array_equal(joint.image, primary.image)
    np.testing.assert_array_equal(joint.channels["flair"], flair.image)
    np.testing.assert_array_equal(joint.mask, primary.mask)
    np.testing.assert_array_equal(joint.affine, primary.affine)
    np.testing.assert_array_equal(joint.keypoints, primary.keypoints)
    np.testing.assert_array_equal(joint.bboxes, primary.bboxes)


def test_image_guided_crop_uses_only_primary_foreground():
    image = np.zeros((8, 9))
    image[-1, -1] = 1
    other = np.zeros_like(image)
    other[0, 0] = 3
    other[-1, -1] = 2
    volume = MedVolume(image, channels={"flair": other})
    out = AnatomicCrop((3, 3), foreground_prob=1, seed=7)(volume)
    assert out.image[-1, -1] == 1
    assert out.channels["flair"][-1, -1] == 2
    assert not (out.channels["flair"] == 3).any()


def test_loading_registration_copy_and_channel_order(registered):
    primary = registered.select_channel()
    flair = registered.select_channel("flair")
    with_extras = primary.with_channels({"flair": flair, "second": primary})
    np.testing.assert_array_equal(with_extras.stack_channels()[1], flair.image)
    np.testing.assert_array_equal(with_extras.stack_channels("second", "flair")[2], flair.image)
    with_extras.copy().channels["flair"][0, 0, 0] = -1
    assert with_extras.channels["flair"][0, 0, 0] != -1
    with pytest.raises(ValueError, match="already exists"):
        with_extras.with_channels({"flair": flair})
    affine = flair.affine.copy()
    affine[0, 3] += 1
    with pytest.raises(ValueError, match="patient-space grid"):
        primary.with_channels({"flair": flair.replace(affine=affine)})
    with pytest.raises(ValueError, match="known patient geometry"):
        primary.with_channels({"flair": MedVolume(flair.image)})
    with pytest.raises(ValueError, match="shape"):
        primary.replace(channels={"flair": np.zeros((9, 12))})
    with pytest.raises(ValueError, match="requires image_map"):
        with_extras.warp(lambda points: points, image=primary.image.copy())


def test_channel_replay_validates_data_and_appearance_keeps_extras(registered):
    policy = Compose([RandomFlip(p_per_axis=1), GaussianNoise(std=0.15)], seed=12)
    out, record = record_augmentation(policy, registered)
    restored = replay_augmentation(record, registered)
    np.testing.assert_array_equal(restored.stack_channels(), out.stack_channels())
    corrupted = registered.copy()
    corrupted.channels["flair"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="fingerprint"):
        replay_augmentation(record, corrupted)
    assert not VolumeValidator().validate(corrupted).ok
    assert not VolumeValidator().validate(registered.select_channel(), registered).ok
    assert volume_summary(corrupted)["channels"]["flair"]["finite_fraction"] < 1
    np.testing.assert_array_equal(
        GaussianNoise(std=0.15, seed=3)(registered).channels["flair"],
        registered.channels["flair"],
    )


def test_volume_state_from_before_named_channels_still_loads(registered):
    state = vars(registered.select_channel()).copy()
    del state["channels"]
    restored = MedVolume.__new__(MedVolume)
    restored.__setstate__(state)
    assert not restored.channels
    np.testing.assert_array_equal(restored.copy().image, registered.image)
