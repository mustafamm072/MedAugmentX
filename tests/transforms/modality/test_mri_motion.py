import numpy as np
import pytest

from medaugmentx.core import MedVolume
from medaugmentx.transforms import MRIMotion


@pytest.fixture
def vol2d():
    img = np.random.default_rng(0).random((48, 48)).astype(np.float32)
    mask = (img > 0.5).astype(np.uint8)
    return MedVolume(image=img, mask=mask)


@pytest.fixture
def vol3d():
    img = np.random.default_rng(1).random((6, 32, 32)).astype(np.float32)
    return MedVolume(image=img)


def test_shape_dtype_preserved_2d(vol2d):
    out = MRIMotion(seed=0)(vol2d)
    assert out.image.shape == vol2d.image.shape
    assert out.image.dtype == np.float32


def test_shape_preserved_3d(vol3d):
    out = MRIMotion(seed=0)(vol3d)
    assert out.image.shape == vol3d.image.shape


def test_changes_image(vol2d):
    out = MRIMotion(degrees=4.0, translation=3.0, seed=0)(vol2d)
    assert not np.allclose(out.image, vol2d.image)


def test_mask_untouched(vol2d):
    out = MRIMotion(seed=0)(vol2d)
    np.testing.assert_array_equal(out.mask, vol2d.mask)


def test_output_finite(vol3d):
    out = MRIMotion(seed=0)(vol3d)
    assert np.isfinite(out.image).all()


def test_deterministic(vol2d):
    a = MRIMotion(degrees=(1.0, 5.0), seed=9)(vol2d)
    b = MRIMotion(degrees=(1.0, 5.0), seed=9)(vol2d)
    np.testing.assert_array_equal(a.image, b.image)


def test_p_zero_no_op(vol2d):
    out = MRIMotion(p=0.0, seed=0)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        MRIMotion(num_movements=(3, 1))
    with pytest.raises(ValueError):
        MRIMotion(degrees=-1.0)


def test_to_dict_round_trip():
    t = MRIMotion(degrees=(1.0, 5.0), translation=(1.0, 4.0), num_movements=(1, 3), p=0.5)
    d = t.to_dict()
    assert d["name"] == "MRIMotion"
    t2 = MRIMotion(**d["params"])
    assert t2.degrees_range == t.degrees_range
    assert t2.num_range == t.num_range


def test_magnitude_ranges_move_in_both_directions():
    image = np.zeros((32, 32), dtype=np.float32)
    image[12:20, 12:20] = 1.0
    volume = MedVolume(image)
    rows = np.arange(32, dtype=np.float64)[:, None]
    shifts = []
    for seed in range(60):
        out = MRIMotion(degrees=(1.0, 5.0), translation=(2.0, 4.0), num_movements=1, seed=seed)(
            volume
        ).image
        shifts.append(float((out * rows).sum() / out.sum()) - 15.5)
    shifts = np.asarray(shifts)
    assert (shifts > 0.1).any() and (shifts < -0.1).any()
    assert abs(shifts.mean()) < 0.5


def test_explicit_signed_range_keeps_its_direction():
    image = np.zeros((32, 32), dtype=np.float32)
    image[12:20, 12:20] = 1.0
    rows = np.arange(32, dtype=np.float64)[:, None]
    for seed in range(10):
        out = MRIMotion(degrees=0.0, translation=(-4.0, -2.0), num_movements=1, seed=seed)(
            MedVolume(image)
        ).image
        assert (out * rows).sum() / out.sum() < 15.5
