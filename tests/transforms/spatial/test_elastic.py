import numpy as np
import pytest

from medaugmentx.core import MedVolume
from medaugmentx.transforms import ElasticDeform


def test_zero_alpha_is_near_identity():
    img = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    vol = MedVolume(image=img)
    out = ElasticDeform(alpha=0.0, sigma=1.0, seed=0)(vol)
    np.testing.assert_allclose(out.image, img, atol=1e-5)


def test_3d_anisotropic_changes_image():
    img = np.random.default_rng(0).random((8, 32, 32)).astype(np.float32)
    vol = MedVolume(image=img)
    out = ElasticDeform(alpha=(20.0, 20.0, 2.0), sigma=(4.0, 4.0, 1.0), seed=0)(vol)
    assert out.image.shape == img.shape
    assert not np.allclose(out.image, img)


def test_mask_label_set_preserved():
    img = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[5:10, 5:10] = 4
    vol = MedVolume(image=img, mask=mask)
    out = ElasticDeform(alpha=10.0, sigma=2.0, seed=1)(vol)
    assert set(np.unique(out.mask).tolist()).issubset({0, 4})
    assert out.mask.dtype == np.uint8


def test_seedable_reproducibility():
    img = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    vol = MedVolume(image=img)
    a = ElasticDeform(alpha=8.0, sigma=2.0, seed=99)(vol)
    b = ElasticDeform(alpha=8.0, sigma=2.0, seed=99)(vol)
    np.testing.assert_array_equal(a.image, b.image)


def test_sigma_controls_spatial_smoothing_for_every_component():
    fields = ElasticDeform(alpha=1, sigma=(0.5, 4, 4), seed=7)._displacements((24, 48, 48))
    for field in fields:
        # Each component should vary faster along z than along x or y.
        roughness = [np.mean(np.diff(field, axis=ax) ** 2) for ax in range(3)]
        assert roughness[0] > 3 * roughness[1]
        assert roughness[0] > 3 * roughness[2]


@pytest.mark.parametrize("shape", [(8, 10), (6, 8, 10)])
@pytest.mark.parametrize("mode", ["constant", "grid-constant"])
def test_constant_fill_keeps_boundary_targets_mappable(monkeypatch, shape, mode):
    """Constant image padding must not make a smooth inverse field discontinuous."""
    shift = np.arange(1, len(shape) + 1, dtype=float)
    fields = [np.full(shape, value, np.float32) for value in shift]
    transform = ElasticDeform(mode=mode, cval=-1000, order=0)
    monkeypatch.setattr(transform, "_displacements", lambda *_: fields)
    points = np.array([np.zeros(len(shape)), np.array(shape) - 1])
    box = np.concatenate([points[0], points[1]])[None]
    volume = MedVolume(
        np.full(shape, 50, np.float32),
        mask=np.ones(shape, np.uint8),
        keypoints=points,
        bboxes=box,
        keypoint_labels=["first", "last"],
        bbox_labels=["organ"],
    )

    out = transform(volume)

    np.testing.assert_allclose(out.keypoints, points - shift)
    assert np.all(out.bboxes[0, : len(shape)] <= out.keypoints[0])
    assert np.all(out.bboxes[0, len(shape) :] >= out.keypoints[1])
    np.testing.assert_array_equal(out.keypoint_labels, volume.keypoint_labels)
    np.testing.assert_array_equal(out.bbox_labels, volume.bbox_labels)
    assert out.image.flat[0] == 50 and out.mask.flat[0] == 1
    assert out.image.flat[-1] == -1000 and out.mask.flat[-1] == 0


def test_conservative_boxes_do_not_require_a_corner_inverse(monkeypatch):
    """Conservative bounds remain available when corner iteration cannot converge."""
    shape = (8, 10)
    yy, xx = np.indices(shape, dtype=np.float32)
    # The unbounded pull field q+d(q)=3q has an inverse, but fixed-point
    # iteration q <- p-d(q) is not contractive. Bounding extrema do not need it.
    fields = [2 * yy, 2 * xx]
    transform = ElasticDeform(max_gradient=None, bbox_mode="conservative")
    monkeypatch.setattr(transform, "_displacements", lambda *_: fields)
    volume = MedVolume(np.ones(shape), bboxes=[[1, 1, 6, 8]], bbox_labels=[7])

    out = transform(volume)

    # The true inverse is q=p/3, so every interior point must be enclosed.
    y, x = np.mgrid[1:7, 1:9]
    mapped = np.column_stack([y.ravel(), x.ravel()]) / 3
    assert np.all(mapped >= out.bboxes[0, :2])
    assert np.all(mapped <= out.bboxes[0, 2:])
    np.testing.assert_array_equal(out.bbox_labels, [7])
