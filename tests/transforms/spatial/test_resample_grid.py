"""Patient-grid resampling regression tests with independent voxel oracles."""

import numpy as np
import pytest

from medaugmentx import MedVolume
from medaugmentx.serialization import from_yaml, to_yaml
from medaugmentx.transforms import ResampleToGrid


def test_reference_grid_controls_axes_extent_fill_and_patient_coordinates():
    pixels = np.arange(4 * 5 * 7, dtype=np.float32).reshape((4, 5, 7))
    mask = pixels.astype(np.uint64) + np.uint64(2**63 + 1)
    source_affine = np.array(
        [[0.1, 0, -0.7, 100], [0, 1.2, 0.1, -20], [2, 0.1, 0, 7], [0, 0, 0, 1]]
    )
    volume = MedVolume(
        pixels,
        mask,
        affine=source_affine,
        keypoints=[[0, 2, 4]],
        bboxes=[[0, 1, 3, 2, 3, 5]],
        bbox_labels=[9],
    )
    pull = np.array([[0, 1, 0, -1], [1, 0, 0, 1], [0, 0, -1, 6], [0, 0, 0, 1]])
    target_affine = source_affine @ pull
    shape = (5, 4, 4)
    resample = ResampleToGrid(shape, target_affine, order=0, cval=-1024, seed=7)
    output = from_yaml(to_yaml(resample))(volume)
    expected_image = np.full(shape, -1024, dtype=np.float32)
    expected_mask = np.zeros(shape, dtype=np.uint64)
    for z, y, x in np.ndindex(shape):
        if 0 <= y - 1 < 4 and 0 <= z + 1 < 5:
            expected_image[z, y, x] = pixels[y - 1, z + 1, 6 - x]
            expected_mask[z, y, x] = mask[y - 1, z + 1, 6 - x]
    np.testing.assert_array_equal(output.image, expected_image)
    np.testing.assert_array_equal(output.mask, expected_mask)
    assert output.mask.dtype == mask.dtype
    np.testing.assert_allclose(output.affine, target_affine, atol=1e-12)
    np.testing.assert_allclose(output.spacing, np.linalg.norm(target_affine[:3, :3], axis=0))
    np.testing.assert_allclose(output.keypoints, [[1, 1, 2]], atol=1e-12)
    np.testing.assert_allclose(
        output.index_to_world(output.keypoints), volume.index_to_world(volume.keypoints)
    )
    np.testing.assert_allclose(output.bboxes, [[0, 1, 1, 2, 3, 3]], atol=1e-12)
    np.testing.assert_array_equal(output.bbox_labels, [9])


def test_coplanar_oblique_2d_grid_with_linear_interpolation():
    affine = np.array([[0, 1, 1, 9], [2, 0, 2, 8], [1, 0.1, 1, 7], [0, 0, 0, 1]])
    image = np.add.outer(10 * np.arange(5), np.arange(6)).astype(np.float32)
    volume = MedVolume(image, affine=affine)
    target = affine.copy()
    target[:3, 3] += 0.25 * affine[:3, 0] + 0.4 * affine[:3, 1]
    target[:3, 2] = [1, 1, -1]
    out = ResampleToGrid((4, 5), target)(volume)
    np.testing.assert_allclose(out.image, image[:4, :5] + 2.9, rtol=1e-6)
    np.testing.assert_allclose(out.affine, target, atol=1e-12)
    assert out.mask is None


def test_reference_grid_rejects_non_coplanar_even_with_same_origin():
    volume = MedVolume(np.zeros((16, 17)), affine=np.eye(4))
    target_affine = np.eye(4)
    target_affine[2, 1] = 0.01
    with pytest.raises(ValueError, match="coplanar"):
        ResampleToGrid(volume.shape, target_affine)(volume)


def test_reference_grid_requires_patient_geometry_and_matching_ndim():
    resample = ResampleToGrid((2, 4), np.eye(4))
    with pytest.raises(ValueError, match="unknown"):
        resample(MedVolume(np.zeros((3, 4))))
    with pytest.raises(ValueError, match="ndim"):
        resample(MedVolume(np.zeros((2, 3, 4)), affine=np.eye(4)))


def test_reference_grid_rejects_unrepresentable_relative_origin():
    source = np.eye(4)
    source[0, 3] = -1e308
    target = np.eye(4)
    target[0, 3] = 1e308
    volume = MedVolume(np.zeros((2, 2, 2)), affine=source)
    with pytest.raises(ValueError, match="unrepresentable"):
        ResampleToGrid(volume.shape, target)(volume)


def test_reference_grid_owns_config_and_uses_identity_and_skip_paths():
    affine = np.eye(4)
    volume = MedVolume(np.zeros((3, 4)), affine=affine)
    resample = ResampleToGrid(volume.shape, affine)
    affine[2, 3] = 5
    assert resample(volume) is volume
    assert ResampleToGrid((1, 1), affine, p=0)(volume) is volume


@pytest.mark.parametrize(
    "params, match",
    [
        ({"shape": (3,)}, "shape"),
        ({"shape": (0, 5)}, "shape"),
        ({"affine": np.zeros((4, 4))}, "affine"),
        ({"cval": 1e100}, "cval"),
        ({"order": 4}, "order"),
        ({"max_voxels": 3}, "max_voxels"),
        ({"shape": (10**10, 10**10)}, "unrepresentable"),
    ],
)
def test_reference_grid_rejects_invalid_settings(params, match):
    with pytest.raises(ValueError, match=match):
        ResampleToGrid(**{"shape": (3, 4), "affine": np.eye(4), **params})
