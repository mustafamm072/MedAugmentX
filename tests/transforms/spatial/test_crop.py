import numpy as np
import pytest

from medaugmentx import apply_for_sample, record_augmentation, replay_augmentation
from medaugmentx.core import MedVolume
from medaugmentx.serialization import from_json, from_yaml, to_json, to_yaml
from medaugmentx.transforms import AnatomicCrop, ClassBalancedCrop


def test_crop_returns_correct_shape():
    img = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    vol = MedVolume(image=img)
    out = AnatomicCrop(size=(16, 16), seed=0)(vol)
    assert out.image.shape == (16, 16)


def test_crop_3d_returns_correct_shape():
    img = np.random.default_rng(0).random((10, 32, 32)).astype(np.float32)
    vol = MedVolume(image=img)
    out = AnatomicCrop(size=(5, 16, 16), seed=0)(vol)
    assert out.image.shape == (5, 16, 16)


def test_crop_size_too_large_clips_to_volume():
    img = np.zeros((8, 8), dtype=np.float32)
    vol = MedVolume(image=img)
    out = AnatomicCrop(size=(32, 32), seed=0)(vol)
    assert out.image.shape == (8, 8)


def test_crop_biases_to_foreground():
    img = np.zeros((32, 32), dtype=np.float32)
    img[28:30, 28:30] = 1.0  # foreground in the corner
    mask = (img > 0).astype(np.uint8)
    vol = MedVolume(image=img, mask=mask)
    # foreground_prob=1.0 -> always biased; the patch must contain a foreground voxel
    out = AnatomicCrop(size=(8, 8), foreground_prob=1.0, seed=0)(vol)
    assert (out.mask > 0).any()


def test_crop_seedable():
    img = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    vol = MedVolume(image=img)
    a = AnatomicCrop(size=(16, 16), seed=42)(vol)
    b = AnatomicCrop(size=(16, 16), seed=42)(vol)
    np.testing.assert_array_equal(a.image, b.image)


def test_size_ndim_must_match():
    vol = MedVolume(image=np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="ndim"):
        AnatomicCrop(size=(4, 4, 4))(vol)


def test_class_balanced_crop_anchors_rare_and_common_labels_equally():
    mask = np.ones((24, 24), dtype=np.uint8)
    mask[-1, -1] = 2
    volume = MedVolume(mask.astype(np.float32), mask)
    crop = ClassBalancedCrop((1, 1), foreground_prob=1, seed=42)
    patches = [crop(volume) for _ in range(300)]
    assert 120 <= sum(patch.mask.item() == 2 for patch in patches) <= 180
    assert all(patch.image.item() == patch.mask.item() for patch in patches)


def test_class_balanced_crop_pads_crops_and_updates_patient_geometry():
    image = np.arange(2 * 6 * 8, dtype=np.int16).reshape(2, 6, 8)
    rare_label = 2**64 - 2
    mask = np.full(image.shape, rare_label + 1, dtype=np.uint64)
    mask[0, 5, 7] = rare_label
    affine = np.array([[0.1, 0, 0.5, 20], [0, -0.8, 0, 5], [2, 0, 0, 1], [0, 0, 0, 1]])
    volume = MedVolume(
        image,
        mask,
        affine=affine,
        keypoints=[[0, 5, 7], [0, 0, 0]],
        keypoint_labels=["rare", "outside"],
        bboxes=[[0, 4, 6, 1, 5, 7]],
        bbox_labels=[3],
    )
    out = ClassBalancedCrop((5, 3, 4), 1, labels=[rare_label, -1], cval=-1024)(volume)

    assert out.shape == (5, 3, 4)
    assert out.spacing == volume.spacing
    assert out.image.dtype == np.float32
    np.testing.assert_array_equal(out.image[1:3], image[:, 3:, 4:])
    np.testing.assert_array_equal(out.image[[0, 3, 4]], -1024)
    assert out.mask.dtype == mask.dtype
    np.testing.assert_array_equal(out.mask[1:3], mask[:, 3:, 4:])
    np.testing.assert_array_equal(out.mask[[0, 3, 4]], 0)
    np.testing.assert_array_equal(out.keypoints, [[1, 2, 3], [1, -3, -4]])
    np.testing.assert_array_equal(out.keypoint_labels, volume.keypoint_labels)
    np.testing.assert_allclose(
        out.index_to_world(out.keypoints), volume.index_to_world(volume.keypoints)
    )
    np.testing.assert_array_equal(out.bboxes, [[1, 1, 2, 2, 2, 3]])
    np.testing.assert_array_equal(out.bbox_labels, [3])
    assert not np.shares_memory(out.image, image)
    assert not np.shares_memory(out.mask, mask)


@pytest.mark.parametrize("label", [None, 0, 1])
def test_class_balanced_crop_falls_back_to_image_uniform_sampling(label):
    image = np.arange(20, dtype=np.int16).reshape(4, 5)
    mask = None if label is None else np.full(image.shape, label, dtype=np.uint8)
    volume = MedVolume(image, mask, spacing=(2, 1))
    crop = ClassBalancedCrop((2, 3), 1, labels=[2], seed=7)
    patches = [crop(volume) for _ in range(20)]
    assert len({patch.image[0, 0] for patch in patches}) > 1
    for patch in patches:
        assert patch.shape == (2, 3)
        assert patch.image.dtype == image.dtype
        assert patch.affine is None
        assert patch.spacing == volume.spacing
        if mask is None:
            assert patch.mask is None
        else:
            np.testing.assert_array_equal(patch.mask, label)


def test_class_balanced_crop_round_trips_and_replays():
    mask = np.zeros((8, 12), dtype=np.int8)
    mask[-2, -4] = -3
    mask[:2, :3] = 1
    volume = MedVolume(mask.astype(np.float32), mask)
    crop = ClassBalancedCrop((9, 4), 1, labels=[-3], cval=-2, seed=5)
    expected, record = record_augmentation(crop, volume)
    assert (expected.mask == -3).any()
    assert not (expected.mask == 1).any()
    np.testing.assert_array_equal(replay_augmentation(record, volume).mask, expected.mask)
    for dump, load in [(to_json, from_json), (to_yaml, from_yaml)]:
        restored = load(dump(crop))
        assert restored.to_dict() == crop.to_dict()
        np.testing.assert_array_equal(restored(volume).image, expected.image)
        before = apply_for_sample(crop, volume, seed=2, sample_id="scan", epoch=3)
        after = apply_for_sample(restored, volume, seed=2, sample_id="scan", epoch=3)
        np.testing.assert_array_equal(before.image, after.image)
    assert ClassBalancedCrop((1, 2), p=0)(volume) is volume


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"size": (3,)}, "size"),
        ({"size": (3, 0)}, "size"),
        ({"labels": [0]}, "labels"),
        ({"labels": [True]}, "labels"),
        ({"labels": []}, "labels"),
        ({"cval": 1e100}, "cval"),
        ({"foreground_prob": np.nan}, "foreground_prob"),
    ],
)
def test_class_balanced_crop_validates_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ClassBalancedCrop(**{"size": (2, 3), **kwargs})


def test_class_balanced_crop_rejects_shape_dimensionality_mismatch():
    with pytest.raises(ValueError, match="ndim"):
        ClassBalancedCrop((3, 4, 5))(MedVolume(np.zeros((9, 11))))
