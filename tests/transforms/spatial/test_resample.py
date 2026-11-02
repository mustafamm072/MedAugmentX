"""Exact spacing, independent sampling oracles, and physical target invariants."""

import json

import numpy as np
import pytest

from medaugmentx import Compose, MedVolume, record_augmentation, replay_augmentation
from medaugmentx.serialization import from_json, from_yaml, to_json, to_yaml
from medaugmentx.transforms import ResampleToSpacing


@pytest.mark.parametrize(
    "shape,spacing,target,expected",
    [
        ((5, 7), (1.2, 0.8), (1.0, 1.5), (6, 4)),
        ((3, 5, 7), (2, 1.2, 0.8), 1.0, (6, 6, 6)),
        ((1, 5), (0.1, 1), (10, 2), (1, 3)),
    ],
)
def test_exact_spacing_and_nearest_size_rounding(shape, spacing, target, expected):
    volume = MedVolume(np.ones(shape, np.int16), spacing=spacing)
    out = ResampleToSpacing(target)(volume)
    assert out.shape == expected
    assert out.spacing == pytest.approx(
        target if isinstance(target, tuple) else (target,) * len(shape)
    )
    assert out.image.dtype == np.float32
    assert out.affine is None
    np.testing.assert_allclose(out.image, 1)


def test_linear_ramp_samples_the_documented_grid():
    y, x = np.mgrid[:7, :9]
    volume = MedVolume((3 * y + 2 * x).astype(np.float32), spacing=(2, 1))
    out = ResampleToSpacing((3, 1.5), antialias=False)(volume)
    assert out.shape == (5, 6)
    qy, qx = np.indices(out.shape)
    # Centres coincide: q=0 samples (-0.0, 0.25) in the input grid.
    expected = 3 * (1.5 * qy) + 2 * (1.5 * qx + 0.25)
    np.testing.assert_allclose(out.image, expected, atol=1e-6)


@pytest.mark.parametrize("shape", [(7, 9), (5, 7, 9)])
def test_sheared_reflected_grid_preserves_world_targets_and_centre(shape):
    ndim = len(shape)
    affine = np.array([[0.3, 0.1, -0.7, 11], [0.2, 1.2, 0, -8], [2, 0, 0.1, 4], [0, 0, 0, 1]])
    volume = MedVolume(
        np.arange(np.prod(shape)).reshape(shape),
        affine=affine,
        keypoints=[np.zeros(ndim), np.asarray(shape) - 1],
        keypoint_labels=["a", "b"],
        bboxes=[[*np.zeros(ndim), *(np.asarray(shape) - 1)]],
        bbox_labels=[4],
    )
    out = ResampleToSpacing(0.9)(volume)
    np.testing.assert_allclose(out.spacing, 0.9)
    np.testing.assert_allclose(
        out.affine[:3, :ndim] / np.asarray(out.spacing),
        volume.affine[:3, :ndim] / np.asarray(volume.spacing),
    )
    np.testing.assert_allclose(
        out.index_to_world(out.keypoints), volume.index_to_world(volume.keypoints), atol=1e-10
    )
    for corner in (slice(None, ndim), slice(ndim, None)):
        np.testing.assert_allclose(
            out.index_to_world(out.bboxes[:, corner]),
            volume.index_to_world(volume.bboxes[:, corner]),
            atol=1e-10,
        )
    np.testing.assert_allclose(
        out.index_to_world([(np.asarray(out.shape) - 1) / 2]),
        volume.index_to_world([(np.asarray(shape) - 1) / 2]),
    )
    np.testing.assert_array_equal(out.keypoint_labels, volume.keypoint_labels)
    np.testing.assert_array_equal(out.bbox_labels, volume.bbox_labels)
    np.testing.assert_array_equal(volume.affine, affine)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32, bool])
def test_image_mask_alignment_and_source_immutability(dtype):
    mask = np.zeros((5, 7, 9), dtype=dtype)
    mask[1:4, 2:6, 2:7] = 1 if dtype is bool else 7
    image = mask.astype(np.float32)
    volume = MedVolume(image.copy(), mask.copy(), spacing=(2, 1.3, 0.8))
    out = ResampleToSpacing(1, order=0, antialias=False)(volume)
    np.testing.assert_array_equal(out.image, out.mask)
    assert out.mask.dtype == mask.dtype
    assert set(np.unique(out.mask)) <= set(np.unique(mask))
    np.testing.assert_array_equal(volume.image, image)
    np.testing.assert_array_equal(volume.mask, mask)


@pytest.mark.parametrize(
    "shape,axis", [((5, 3), 0), ((5, 3), 1), ((3, 5, 7), 0), ((3, 5, 7), 1), ((3, 5, 7), 2)]
)
@pytest.mark.parametrize("spacing", [0.6, 1.4, 2.0])
def test_nearest_neighbor_alignment_at_half_voxel_boundaries(shape, axis, spacing):
    image = np.indices(shape)[axis].astype(np.float32)
    mask = image.astype(np.int16)
    target = [1.0] * len(shape)
    target[axis] = spacing
    out = ResampleToSpacing(target, order=0, antialias=False)(MedVolume(image, mask))
    np.testing.assert_array_equal(out.image, out.mask)


@pytest.mark.parametrize("dtype,label", [(np.int64, 2**60 + 3), (np.uint64, 2**63 + 3)])
def test_half_voxel_boundaries_keep_large_mask_ids_aligned(dtype, label):
    image = np.repeat(np.arange(5)[:, None], 3, axis=1).astype(np.float32)
    mask = np.zeros(image.shape, dtype=dtype)
    mask[image == 4] = label
    out = ResampleToSpacing((0.6, 1), order=0, antialias=False)(MedVolume(image, mask))
    np.testing.assert_array_equal(out.image == 4, out.mask == label)
    assert {int(value) for value in np.unique(out.mask)} == {0, label}


def test_rounded_shape_can_stay_same_while_spacing_changes():
    volume = MedVolume(np.arange(20).reshape(4, 5), spacing=(1, 1), affine=np.eye(4))
    out = ResampleToSpacing(1.01)(volume)
    assert out.shape == volume.shape
    np.testing.assert_allclose(out.spacing, 1.01)
    assert not np.array_equal(out.image, volume.image)


@pytest.mark.parametrize("dtype,label", [(np.int64, 2**60 + 3), (np.uint64, 2**63 + 3)])
def test_large_integer_label_ids_remain_exact(dtype, label):
    mask = np.zeros((4, 5), dtype=dtype)
    mask[1:3, 1:4] = label
    out = ResampleToSpacing(0.6)(MedVolume(np.ones((4, 5)), mask=mask))
    assert out.mask.dtype == dtype
    assert {int(v) for v in np.unique(out.mask)} == {0, label}


def test_identity_and_probability_gate():
    volume = MedVolume(np.ones((4, 5)), spacing=(2, 1))
    assert ResampleToSpacing((2, 1))(volume) is volume
    assert ResampleToSpacing(0.1, p=0)(volume) is volume


def test_antialias_reduces_high_frequency_energy():
    y, x = np.mgrid[:64, :64]
    volume = MedVolume(np.cos(x * 2 * np.pi / 3) + np.cos(y * 2 * np.pi / 3))
    filtered = ResampleToSpacing(4)(volume).image
    aliased = ResampleToSpacing(4, antialias=False)(volume).image
    assert filtered.var() < aliased.var() * 0.1


def test_clip_output_bounds_cubic_overshoot():
    image = np.zeros((8, 8), np.float32)
    image[2:6, 2:6] = 1
    volume = MedVolume(image)
    raw = ResampleToSpacing(0.6, order=3)(volume).image
    assert raw.min() < 0 or raw.max() > 1
    clipped = ResampleToSpacing(0.6, order=3, clip_output=True)(volume).image
    assert clipped.min() >= 0 and clipped.max() <= 1


@pytest.mark.parametrize(
    "spacing", [0, -1, np.nan, np.inf, True, (), (1, 0), (1, np.inf), (1, "2")]
)
def test_invalid_spacing(spacing):
    with pytest.raises(ValueError, match="spacing"):
        ResampleToSpacing(spacing)


@pytest.mark.parametrize("order", [-1, 4, 1.5, True])
def test_invalid_interpolation(order):
    with pytest.raises(ValueError, match="order"):
        ResampleToSpacing(1, order=order)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_voxel_limit(limit):
    with pytest.raises(ValueError, match="max_voxels"):
        ResampleToSpacing(1, max_voxels=limit)


def test_output_limit_rejects_before_resampling(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Allocation/resampling must not start before the size check")

    monkeypatch.setattr("medaugmentx.transforms.spatial.resample.affine_transform", unexpected)
    monkeypatch.setattr("medaugmentx.transforms.spatial.resample.gaussian_filter", unexpected)
    volume = MedVolume(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="max_voxels"):
        ResampleToSpacing(0.01, max_voxels=100)(volume)


def test_dimension_mismatch():
    with pytest.raises(ValueError, match="one entry"):
        ResampleToSpacing((1, 2, 3))(MedVolume(np.zeros((3, 4))))


@pytest.mark.parametrize("spacing", [1e-300, 1e300])
def test_unrepresentable_grid(spacing):
    volume = MedVolume(np.zeros((3, 4)), spacing=(1e100, 1e-100))
    with pytest.raises(ValueError, match="unrepresentable"):
        ResampleToSpacing(spacing)(volume)


def test_finite_grid_too_large_for_platform_indices():
    with pytest.raises(ValueError, match="unrepresentable"):
        ResampleToSpacing(1e-12)(MedVolume(np.zeros((3, 4))))


@pytest.mark.parametrize("spacing", [0.7, (1.3, 0.7)])
@pytest.mark.parametrize("dump,load", [(to_json, from_json), (to_yaml, from_yaml)])
def test_serialization_and_record_replay(spacing, dump, load):
    if dump is to_yaml:
        pytest.importorskip("yaml")
    policy = Compose(
        [ResampleToSpacing(spacing, order=3, clip_output=True, max_voxels=1000)], seed=4
    )
    restored = load(dump(policy))
    assert restored.to_dict() == policy.to_dict()
    volume = MedVolume(np.arange(30).reshape(5, 6), affine=np.eye(4))
    expected, record = record_augmentation(restored, volume)
    actual = replay_augmentation(json.loads(json.dumps(record)), volume)
    np.testing.assert_array_equal(actual.image, expected.image)
    np.testing.assert_array_equal(actual.affine, expected.affine)
