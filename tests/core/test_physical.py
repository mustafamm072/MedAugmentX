"""Independent physical-coordinate invariants for known and unknown grids."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from medaugmentx import MedVolume
from medaugmentx.transforms import AnatomicCrop, CenterCrop, Pad, Resize


def _volume(shape=(10, 12, 14)):
    affine = np.array([[0.3, 0.1, 0.7, 11], [0.2, 1.2, 0, -8], [2, 0, 0.1, 4], [0, 0, 0, 1]])
    return MedVolume(np.zeros(shape, np.float32), affine=affine, keypoints=[[5, 6, 7]])


@pytest.mark.parametrize(
    "transform",
    [
        CenterCrop((4, 4, 4)),
        Pad((14, 18, 20)),
        Resize((7, 17, 21)),
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
