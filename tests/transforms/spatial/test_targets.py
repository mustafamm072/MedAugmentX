"""Keypoints and bounding boxes are warped in lockstep with the image.

The recurring strategy: place a single bright pixel at a keypoint, run the
transform, and assert the keypoint still coincides with the (moved) pixel peak.
Deterministic transforms are also checked against exact expected coordinates.
"""
import numpy as np

from medaugmentx.core import Compose, MedVolume
from medaugmentx.transforms import (
    AnatomicCrop,
    BrightnessContrast,
    CenterCrop,
    CoarseDropout,
    ElasticDeform,
    GammaCorrection,
    GaussianNoise,
    Pad,
    RandomAffine,
    RandomFlip,
    Resize,
)


def _peak(image: np.ndarray) -> tuple[int, ...]:
    return np.unravel_index(int(np.argmax(image)), image.shape)


def _volume_with_point(shape, coord):
    img = np.zeros(shape, dtype=np.float32)
    img[tuple(coord)] = 1.0
    return MedVolume(
        image=img,
        keypoints=np.array([[float(c) for c in coord]]),
        keypoint_labels=np.array([42]),
        bboxes=np.array([[*(float(c) for c in coord), *(float(c) + 1 for c in coord)]]),
        bbox_labels=np.array([1]),
    )


class TestFlip:
    def test_flip_x_moves_keypoint_and_bbox(self):
        v = _volume_with_point((10, 20), (3, 5))
        out = RandomFlip(axes=("x",), p_per_axis=1.0, seed=0).apply(v)
        np.testing.assert_allclose(out.keypoints[0], [3.0, 14.0])
        # bbox [3,5,4,6] reflects on x (extent 20): x=5->14, x=6->13, so the
        # re-bounded box is [3, 13, 4, 14].
        np.testing.assert_allclose(out.bboxes[0], [3.0, 13.0, 4.0, 14.0])
        assert out.keypoint_labels.tolist() == [42]
        assert _peak(out.image) == (3, 14)

    def test_no_flip_preserves_targets(self):
        v = _volume_with_point((10, 10), (2, 2))
        out = RandomFlip(axes=("x",), p_per_axis=0.0, seed=0).apply(v)
        np.testing.assert_array_equal(out.keypoints, v.keypoints)


class TestAffine:
    def test_identity_keeps_keypoint(self):
        v = _volume_with_point((21, 21), (5, 8))
        out = RandomAffine(rotation=0.0, p=1.0, seed=0).apply(v)
        np.testing.assert_allclose(out.keypoints[0], [5.0, 8.0], atol=1e-6)

    def test_rotation_keypoint_tracks_pixel(self):
        v = _volume_with_point((41, 41), (10, 30))
        out = RandomAffine(rotation=(25.0, 25.0), order=1, p=1.0, seed=1).apply(v)
        peak = _peak(out.image)
        kp = out.keypoints[0]
        assert abs(peak[0] - kp[0]) <= 1.0
        assert abs(peak[1] - kp[1]) <= 1.0

    def test_3d_rotation_tracks_pixel(self):
        v = _volume_with_point((16, 24, 24), (5, 6, 18))
        out = RandomAffine(rotation=(20.0, 20.0), axes_enabled=("x", "y"), order=1, p=1.0, seed=2).apply(v)
        peak = _peak(out.image)
        kp = out.keypoints[0]
        assert all(abs(p - k) <= 1.5 for p, k in zip(peak, kp))


class TestTranslateFamily:
    def test_pad_shifts_by_before(self):
        v = _volume_with_point((10, 20), (3, 5))
        out = Pad(size=(14, 28)).apply(v)  # before = (2, 4)
        np.testing.assert_allclose(out.keypoints[0], [5.0, 9.0])
        np.testing.assert_allclose(out.bboxes[0], [5.0, 9.0, 6.0, 10.0])
        assert _peak(out.image) == (5, 9)

    def test_center_crop_shifts_negative(self):
        v = _volume_with_point((10, 10), (5, 5))
        out = CenterCrop(size=(4, 4)).apply(v)  # start = (3, 3)
        np.testing.assert_allclose(out.keypoints[0], [2.0, 2.0])
        assert _peak(out.image) == (2, 2)

    def test_anatomic_crop_tracks_pixel(self):
        v = _volume_with_point((16, 16), (7, 9))
        out = AnatomicCrop(size=(8, 8), foreground_prob=1.0, seed=3).apply(v)
        # foreground-biased crop keeps the bright pixel inside the patch.
        peak = _peak(out.image)
        np.testing.assert_allclose(out.keypoints[0], peak, atol=1e-6)


class TestResize:
    def test_resize_scales_coords(self):
        v = _volume_with_point((10, 20), (3, 5))
        out = Resize(size=(20, 40)).apply(v)  # factor (2, 2)
        np.testing.assert_allclose(out.keypoints[0], [6.0, 10.0])
        np.testing.assert_allclose(out.bboxes[0], [6.0, 10.0, 8.0, 12.0])

    def test_resize_same_size_is_noop(self):
        v = _volume_with_point((8, 8), (2, 2))
        out = Resize(size=(8, 8)).apply(v)
        np.testing.assert_array_equal(out.keypoints, v.keypoints)


class TestElastic:
    def test_elastic_keypoint_tracks_pixel(self):
        v = _volume_with_point((60, 60), (20, 40))
        out = ElasticDeform(alpha=8.0, sigma=6.0, order=1, p=1.0, seed=3).apply(v)
        peak = _peak(out.image)
        kp = out.keypoints[0]
        assert abs(peak[0] - kp[0]) <= 2.0
        assert abs(peak[1] - kp[1]) <= 2.0


class TestNonSpatialPassthrough:
    def test_intensity_transforms_preserve_targets(self):
        v = _volume_with_point((16, 16), (4, 4))
        for t in (
            GaussianNoise(std=0.1, seed=0),
            GammaCorrection(gamma=(0.8, 1.2), seed=0),
            BrightnessContrast(seed=0),
        ):
            out = t.apply(v)
            np.testing.assert_array_equal(out.keypoints, v.keypoints)
            np.testing.assert_array_equal(out.bboxes, v.bboxes)

    def test_coarse_dropout_preserves_targets(self):
        v = _volume_with_point((16, 16), (4, 4))
        out = CoarseDropout(num_holes=3, p=1.0, seed=1).apply(v)
        np.testing.assert_array_equal(out.keypoints, v.keypoints)
        np.testing.assert_array_equal(out.bboxes, v.bboxes)


class TestCompose:
    def test_pipeline_threads_targets(self):
        v = _volume_with_point((10, 20), (3, 5))
        pipe = Compose(
            [RandomFlip(axes=("x",), p_per_axis=1.0), Pad(size=(14, 28))],
            seed=0,
        )
        out = pipe(v)
        # flip x: (3, 14); pad before (2, 4): (5, 18)
        np.testing.assert_allclose(out.keypoints[0], [5.0, 18.0])
        assert out.keypoint_labels.tolist() == [42]
