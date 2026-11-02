"""Independent physical-coordinate invariants for known and unknown grids."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from medaugmentx import MedVolume
from medaugmentx.transforms import AnatomicCrop, CenterCrop, Pad, ResampleToSpacing, Resize


def _volume(shape=(10, 12, 14)):
    affine = np.array([[0.3, 0.1, 0.7, 11], [0.2, 1.2, 0, -8], [2, 0, 0.1, 4], [0, 0, 0, 1]])
    return MedVolume(
        np.zeros(shape, np.float32), affine=affine, keypoints=[[5, 6, 7][: len(shape)]]
    )


@pytest.mark.parametrize(
    "transform",
    [
        CenterCrop((4, 4, 4)),
        Pad((14, 18, 20)),
        Resize((7, 17, 21)),
        ResampleToSpacing((1.3, 0.7, 0.9)),
        AnatomicCrop((5, 7, 9), seed=3),
    ],
)
def test_grid_changes_preserve_world_landmarks(transform):
    volume = _volume()
    out = transform(volume)
    np.testing.assert_allclose(
        out.index_to_world(out.keypoints), volume.index_to_world(volume.keypoints), atol=1e-8
    )
    np.testing.assert_allclose(out.spacing, np.linalg.norm(out.affine[:3, :3], axis=0))
    np.testing.assert_allclose(volume.affine[:3, 3], [11, -8, 4])


@given(st.integers(1, 12), st.integers(1, 15))
def test_resize_preserves_voxel_extent_even_for_singletons(old, new):
    affine = np.array([[0, 1, 0, 2], [2, 0, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]], float)
    volume = MedVolume(np.zeros((old, 8)), affine=affine)
    out = Resize((new, 8))(volume)
    before = volume.index_to_world([[-0.5, 0], [old - 0.5, 0]])
    after = out.index_to_world([[-0.5, 0], [new - 0.5, 0]])
    np.testing.assert_allclose(before, after, atol=1e-9)


def test_unknown_geometry_is_explicit():
    volume = MedVolume(np.zeros((4, 4)))
    assert volume.affine is None
    with pytest.raises(ValueError, match="unknown"):
        volume.index_to_world([[1, 1]])


def test_spacing_must_agree_with_affine():
    with pytest.raises(ValueError, match="agree"):
        MedVolume(np.zeros((4, 4)), spacing=(2, 2), affine=np.eye(4))


@pytest.mark.parametrize("shape", [(6, 7), (5, 6, 7)])
def test_world_index_round_trip_on_oblique_sheared_grids(shape):
    volume = _volume(shape)
    points = np.array([np.zeros(len(shape)), [1.25] * len(shape), [-3.5] * len(shape)])
    np.testing.assert_allclose(
        volume.world_to_index(volume.index_to_world(points)), points, atol=1e-12
    )
    assert volume.world_to_index(np.empty((0, 3))).shape == (0, len(shape))


def test_2d_world_to_index_checks_distance_from_actual_plane():
    volume = _volume((6, 7))
    basis = volume.affine[:3, :2]
    normal = np.cross(basis[:, 0], basis[:, 1])
    normal /= np.linalg.norm(normal)
    points = volume.index_to_world([[2, 3]])
    with pytest.raises(ValueError, match="plane"):
        volume.world_to_index(points + normal * 0.01)
    np.testing.assert_allclose(volume.world_to_index(points + normal * 1e-6), [[2, 3]])
    np.testing.assert_allclose(
        volume.world_to_index(points + normal * 0.01, tolerance_mm=0.02), [[2, 3]]
    )


@pytest.mark.parametrize("points", [[[1, 2]], [[1, 2, np.nan]], [1, 2, 3]])
def test_world_to_index_rejects_invalid_points(points):
    with pytest.raises(ValueError):
        _volume().world_to_index(points)


@pytest.mark.parametrize("tolerance", [-1, np.nan, np.inf])
def test_world_to_index_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance_mm"):
        _volume().world_to_index([[0, 0, 0]], tolerance_mm=tolerance)


def test_world_to_index_requires_known_geometry():
    with pytest.raises(ValueError, match="unknown"):
        MedVolume(np.zeros((4, 4))).world_to_index([[1, 1, 0]])
