import numpy as np
import pytest

from medaugmentx import Guard, MedVolume, Transform, VolumeValidator
from medaugmentx.core.geometry import displacement_jacobian
from medaugmentx.serialization import from_json, to_json
from medaugmentx.transforms import (
    AnatomicCrop,
    BiasField,
    BrightnessContrast,
    ElasticDeform,
    GaussianNoise,
    Pad,
    RandomAffine,
    Resize,
    RicianNoise,
    WindowLevel,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GaussianNoise(std=np.nan),
        lambda: RandomAffine(scale=(0, 1)),
        lambda: RandomAffine(scale=(2, 1)),
        lambda: BrightnessContrast(clip=(1, 0)),
        lambda: BiasField(coarse_shape=0),
        lambda: AnatomicCrop(size=(0, 4)),
        lambda: Resize((2.5, 4)),
        lambda: Pad((2, -4)),
        lambda: ElasticDeform(sigma=-1),
        lambda: ElasticDeform(alpha=(1, 2, np.inf)),
        lambda: WindowLevel(width_scale=np.inf),
    ],
)
def test_constructor_domains_fail_early(factory):
    with pytest.raises(ValueError):
        factory()


def test_extreme_elastic_fields_are_gradient_limited_and_boxes_conservative():
    transform = ElasticDeform(alpha=100, sigma=0.5, min_jacobian=0.05, seed=3)
    fields = transform._displacements((12, 14))
    assert displacement_jacobian(fields).min() > 0.05
    volume = MedVolume(np.ones((12, 14)), bboxes=[[2, 3, 9, 11]])
    out = ElasticDeform(alpha=100, sigma=0.5, seed=3)(volume)
    # Bounds include every forward-mapped interior point, not only corners.
    from medaugmentx.core.geometry import displacement_map

    y, x = np.mgrid[2:10, 3:12]
    points = displacement_map(fields)(np.column_stack([y.ravel(), x.ravel()]))
    assert np.all(points >= out.bboxes[0, :2] - 1e-5)
    assert np.all(points <= out.bboxes[0, 2:] + 1e-5)


def test_physical_rotation_is_rigid_on_sheared_grid():
    affine = np.array([[0.2, 0, 1, 2], [0, 2, 0.1, 3], [3, 0.2, 0, 4], [0, 0, 0, 1.0]])
    points = np.array([[4, 5, 6], [6, 7, 7]], float)
    volume = MedVolume(np.zeros((15, 17, 19)), affine=affine, keypoints=points)
    out = RandomAffine(rotation=(20, 20), scale=(1, 1), space="physical", seed=2)(volume)
    before = np.linalg.norm(np.diff(volume.index_to_world(points), axis=0))
    after = np.linalg.norm(np.diff(out.index_to_world(out.keypoints), axis=0))
    assert after == pytest.approx(before, rel=1e-8)


def test_mm_translation_and_elastic_cap():
    volume = MedVolume(np.zeros((20, 20)), spacing=(2, 0.5), keypoints=[[10, 10]])
    out = RandomAffine(translation=(1, 1), translation_units="mm")(volume)
    np.testing.assert_allclose(out.keypoints, [[10.5, 12]])
    fields = ElasticDeform(alpha=100, sigma=1, units="mm", max_displacement=0.2)._displacements(
        volume.shape, volume.spacing
    )
    assert max(float(np.max(np.abs(f))) * s for f, s in zip(fields, volume.spacing)) <= 0.200001


def test_physical_foreground_loss_is_per_class_and_resolution_aware():
    mask = np.ones((16, 16), np.uint8)
    mask[4:8, 4:8] = 2
    reference = MedVolume(np.arange(256, dtype=float).reshape(16, 16), mask=mask)
    down = Resize((8, 8), order=0)(reference)
    validator = VolumeValidator(max_foreground_loss=0.01)
    assert validator.validate(down, reference).ok
    mutated = reference.copy()
    mutated.mask[mutated.mask == 2] = 1
    checks = {issue.check for issue in validator.validate(mutated, reference).errors}
    assert checks == {"mask_labels", "foreground_loss"}


def test_validator_checks_mutated_geometry_and_masks():
    volume = MedVolume(np.arange(16).reshape(4, 4), mask=np.zeros((4, 4)))
    volume.mask[0, 0] = np.nan
    volume.spacing = (-1, 1)
    checks = {issue.check for issue in VolumeValidator().validate(volume).errors}
    assert {"mask_values", "geometry"} <= checks


class FailingDraw(Transform):
    def apply(self, volume):
        volume.image[:] = -999
        raise ValueError("unusable field")


def test_guard_exception_policy_is_explicit_and_measured():
    volume = MedVolume(np.arange(16, dtype=float).reshape(4, 4))
    with pytest.raises(ValueError, match="unusable"):
        Guard(FailingDraw())(volume)
    guard = Guard(FailingDraw(), on_fail="retry", retries=3, catch_errors=True)
    with pytest.warns(UserWarning, match="exhausted"):
        out = guard(volume)
    np.testing.assert_array_equal(out.image, volume.image)
    assert guard.stats["attempts"] == guard.stats["exceptions"] == 3
    assert guard.stats["rejection_rate"] == guard.stats["revert_rate"] == 1
    assert len(guard.last_reports) == 3
    guard.reset_stats()
    assert guard.stats["calls"] == 0


def test_rician_large_amplitudes_do_not_overflow_from_squaring():
    volume = MedVolume(np.full((4, 4), 1e30, np.float32))
    out = RicianNoise(std=0)(volume)
    assert np.isfinite(out.image).all()


def test_histogram_policy_stores_summary_instead_of_image():
    from medaugmentx.transforms import HistogramMatch

    transform = HistogramMatch(reference=np.arange(10_000).reshape(100, 100), n_quantiles=32)
    document = to_json(transform)
    assert len(document) < 5000
    volume = MedVolume(np.arange(64, dtype=np.float32).reshape(8, 8))
    np.testing.assert_array_equal(transform(volume).image, from_json(document)(volume).image)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RandomAffine(rotation=(35, 35), order=0, mode="reflect", seed=2),
        lambda: ElasticDeform(alpha=40, sigma=1, order=0, mode="reflect", seed=2),
    ],
)
def test_reflected_boundary_context_preserves_segmentation_alignment(factory):
    image = np.indices((18, 20)).sum(axis=0) % 2 + 1
    volume = MedVolume(image.astype(np.float32), mask=image.astype(np.uint8))
    out = factory()(volume)
    np.testing.assert_array_equal(out.image, out.mask)
