"""Tests for keypoint/bbox targets on MedVolume and the geometry helpers."""
import numpy as np
import pytest

from medaugmentx.core import MedVolume, geometry

# ---------------------------------------------------------------------------
# MedVolume construction & validation
# ---------------------------------------------------------------------------


class TestTargetConstruction:
    def test_keypoints_and_bboxes_2d(self):
        v = MedVolume(
            image=np.zeros((8, 8), dtype=np.float32),
            keypoints=[[1.0, 2.0], [3.0, 4.0]],
            bboxes=[[0.0, 0.0, 4.0, 4.0]],
        )
        assert v.has_keypoints and v.has_bboxes
        assert v.num_keypoints == 2 and v.num_bboxes == 1
        assert v.keypoints.dtype == np.float64
        assert v.keypoints.shape == (2, 2)
        assert v.bboxes.shape == (1, 4)

    def test_keypoints_3d(self):
        v = MedVolume(
            image=np.zeros((4, 8, 8), dtype=np.float32),
            keypoints=[[1.0, 2.0, 3.0]],
            bboxes=[[0, 0, 0, 2, 2, 2]],
        )
        assert v.keypoints.shape == (1, 3)
        assert v.bboxes.shape == (1, 6)

    def test_labels_ride_along(self):
        v = MedVolume(
            image=np.zeros((8, 8), dtype=np.float32),
            keypoints=[[1.0, 2.0]],
            keypoint_labels=[7],
            bboxes=[[0.0, 0.0, 4.0, 4.0]],
            bbox_labels=["lesion"],
        )
        assert v.keypoint_labels.tolist() == [7]
        assert v.bbox_labels.tolist() == ["lesion"]

    def test_rejects_wrong_keypoint_ndim(self):
        with pytest.raises(ValueError, match="keypoints must have shape"):
            MedVolume(image=np.zeros((8, 8), dtype=np.float32), keypoints=[[1.0, 2.0, 3.0]])

    def test_rejects_wrong_bbox_width(self):
        with pytest.raises(ValueError, match="bboxes must have shape"):
            MedVolume(image=np.zeros((8, 8), dtype=np.float32), bboxes=[[0.0, 0.0, 4.0]])

    def test_rejects_min_greater_than_max(self):
        with pytest.raises(ValueError, match="min <= max"):
            MedVolume(image=np.zeros((8, 8), dtype=np.float32), bboxes=[[5.0, 0.0, 1.0, 4.0]])

    def test_rejects_label_count_mismatch(self):
        with pytest.raises(ValueError, match="keypoint_labels must be 1D"):
            MedVolume(
                image=np.zeros((8, 8), dtype=np.float32),
                keypoints=[[1.0, 2.0]],
                keypoint_labels=[1, 2],
            )

    def test_rejects_labels_without_coords(self):
        with pytest.raises(ValueError, match="keypoint_labels given without"):
            MedVolume(image=np.zeros((8, 8), dtype=np.float32), keypoint_labels=[1])

    def test_empty_targets_allowed(self):
        v = MedVolume(
            image=np.zeros((8, 8), dtype=np.float32),
            keypoints=np.empty((0, 2)),
            bboxes=np.empty((0, 4)),
        )
        assert v.num_keypoints == 0 and v.num_bboxes == 0


class TestReplaceCopy:
    def test_replace_preserves_targets(self):
        v = MedVolume(
            image=np.zeros((8, 8), dtype=np.float32),
            keypoints=[[1.0, 2.0]],
            keypoint_labels=[9],
            bboxes=[[0.0, 0.0, 4.0, 4.0]],
        )
        v2 = v.replace(image=np.ones((8, 8), dtype=np.float32))
        np.testing.assert_array_equal(v2.keypoints, v.keypoints)
        np.testing.assert_array_equal(v2.keypoint_labels, v.keypoint_labels)
        np.testing.assert_array_equal(v2.bboxes, v.bboxes)

    def test_copy_is_deep(self):
        v = MedVolume(
            image=np.zeros((8, 8), dtype=np.float32),
            keypoints=[[1.0, 2.0]],
            bboxes=[[0.0, 0.0, 4.0, 4.0]],
        )
        c = v.copy()
        c.keypoints[0, 0] = 99
        assert v.keypoints[0, 0] == 1.0

    def test_repr_mentions_target_counts(self):
        v = MedVolume(
            image=np.zeros((8, 8), dtype=np.float32),
            keypoints=[[1.0, 2.0]],
            bboxes=[[0.0, 0.0, 4.0, 4.0]],
        )
        s = repr(v)
        assert "keypoints=1" in s and "bboxes=1" in s


class TestWarp:
    def test_warp_no_targets_is_replace(self):
        v = MedVolume(image=np.zeros((4, 4), dtype=np.float32))
        out = v.warp(geometry.translate_map(np.array([1.0, 1.0])), image=np.ones((4, 4), np.float32))
        assert out.keypoints is None
        assert np.all(out.image == 1.0)

    def test_warp_maps_keypoints_and_bboxes(self):
        v = MedVolume(
            image=np.zeros((8, 8), dtype=np.float32),
            keypoints=[[2.0, 3.0]],
            bboxes=[[1.0, 1.0, 3.0, 3.0]],
        )
        out = v.warp(
            geometry.translate_map(np.array([1.0, 2.0])),
            image=np.zeros((8, 8), dtype=np.float32),
        )
        np.testing.assert_allclose(out.keypoints, [[3.0, 5.0]])
        np.testing.assert_allclose(out.bboxes, [[2.0, 3.0, 4.0, 5.0]])


class TestRemoveOutOfBounds:
    def test_drops_out_of_frame_keypoints_with_labels(self):
        v = MedVolume(
            image=np.zeros((10, 10), dtype=np.float32),
            keypoints=[[-1.0, 5.0], [5.0, 5.0], [5.0, 12.0]],
            keypoint_labels=[0, 1, 2],
        )
        out = v.remove_out_of_bounds_targets()
        np.testing.assert_allclose(out.keypoints, [[5.0, 5.0]])
        assert out.keypoint_labels.tolist() == [1]

    def test_clips_bboxes_and_drops_fully_outside(self):
        v = MedVolume(
            image=np.zeros((10, 10), dtype=np.float32),
            bboxes=[[-5.0, -5.0, 3.0, 3.0], [20.0, 20.0, 25.0, 25.0]],
            bbox_labels=["keep", "drop"],
        )
        out = v.remove_out_of_bounds_targets()
        assert out.num_bboxes == 1
        np.testing.assert_allclose(out.bboxes, [[0.0, 0.0, 3.0, 3.0]])
        assert out.bbox_labels.tolist() == ["keep"]

    def test_min_visibility_threshold(self):
        # Box [y=5..20, x=0..9] in a 10x10 image (valid index range 0..9).
        v = MedVolume(
            image=np.zeros((10, 10), dtype=np.float32),
            bboxes=[[5.0, 0.0, 20.0, 9.0]],
        )
        # Original area = 15*9; clipped on y to 5..9 -> height 4 -> area 4*9. frac ~0.27.
        kept = v.remove_out_of_bounds_targets(min_visibility=0.2)
        assert kept.num_bboxes == 1
        dropped = v.remove_out_of_bounds_targets(min_visibility=0.5)
        assert dropped.num_bboxes == 0

    def test_rejects_bad_min_visibility(self):
        v = MedVolume(image=np.zeros((10, 10), dtype=np.float32), keypoints=[[1.0, 1.0]])
        with pytest.raises(ValueError, match="min_visibility"):
            v.remove_out_of_bounds_targets(min_visibility=1.5)


# ---------------------------------------------------------------------------
# geometry pure functions
# ---------------------------------------------------------------------------


class TestGeometryHelpers:
    def test_flip_map(self):
        fn = geometry.flip_map((1,), (10, 20))
        out = fn(np.array([[3.0, 5.0]]))
        np.testing.assert_allclose(out, [[3.0, 14.0]])

    def test_map_bboxes_rebounds_under_rotation(self):
        # 90-degree rotation of a box about the origin should stay a valid AABB.
        boxes = np.array([[0.0, 0.0, 2.0, 4.0]])
        forward = np.array([[0.0, -1.0], [1.0, 0.0]])  # rotate +90deg
        centre = np.array([0.0, 0.0])
        fn = geometry.affine_map(forward, centre, np.array([0.0, 0.0]))
        out = geometry.map_bboxes(boxes, 2, fn)
        mins, maxs = out[0, :2], out[0, 2:]
        assert np.all(maxs >= mins)

    def test_map_keypoints_empty(self):
        out = geometry.map_keypoints(np.empty((0, 2)), geometry.translate_map(np.array([1.0, 1.0])))
        assert out.shape == (0, 2)

    def test_as_bboxes_validates(self):
        with pytest.raises(ValueError):
            geometry.as_bboxes([[0, 0, 1]], 2)
