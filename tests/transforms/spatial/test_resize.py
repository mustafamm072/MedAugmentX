import numpy as np
import pytest

from medaugmentx.core import MedVolume
from medaugmentx.transforms import CenterCrop, Pad, Resize


@pytest.fixture
def vol2d():
    img = np.random.default_rng(0).random((40, 60)).astype(np.float32)
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[10:30, 20:40] = 3
    return MedVolume(image=img, mask=mask, spacing=(0.8, 0.5))


@pytest.fixture
def vol3d():
    img = np.random.default_rng(1).random((12, 32, 32)).astype(np.float32)
    mask = np.zeros((12, 32, 32), dtype=np.uint8)
    mask[3:9, 8:24, 8:24] = 1
    return MedVolume(image=img, mask=mask, spacing=(1.0, 0.7, 0.7))


# ---- Resize ----

def test_resize_to_target_shape_2d(vol2d):
    out = Resize(size=(20, 30))(vol2d)
    assert out.image.shape == (20, 30)
    assert out.mask.shape == (20, 30)
    assert out.image.dtype == np.float32


def test_resize_to_target_shape_3d(vol3d):
    out = Resize(size=(6, 16, 16))(vol3d)
    assert out.image.shape == (6, 16, 16)


def test_resize_rescales_spacing(vol2d):
    out = Resize(size=(20, 30))(vol2d)
    # halving rows doubles row spacing
    assert out.spacing[0] == pytest.approx(0.8 * 40 / 20)
    assert out.spacing[1] == pytest.approx(0.5 * 60 / 30)


def test_resize_mask_dtype_and_labels_preserved(vol3d):
    out = Resize(size=(8, 20, 20))(vol3d)
    assert out.mask.dtype == vol3d.mask.dtype
    assert set(np.unique(out.mask).tolist()) <= {0, 1}


def test_resize_identity_when_same_shape(vol2d):
    out = Resize(size=vol2d.shape)(vol2d)
    np.testing.assert_array_equal(out.image, vol2d.image)


def test_resize_ndim_mismatch_raises(vol2d):
    with pytest.raises(ValueError):
        Resize(size=(10, 10, 10))(vol2d)


def test_resize_to_dict_round_trip():
    t = Resize(size=(16, 16), order=1, p=1.0)
    t2 = Resize(**t.to_dict()["params"])
    assert t2.size == t.size


# ---- Pad ----

def test_pad_grows_to_target(vol2d):
    out = Pad(size=(50, 70))(vol2d)
    assert out.image.shape == (50, 70)
    assert out.mask.shape == (50, 70)


def test_pad_never_crops(vol2d):
    out = Pad(size=(10, 10))(vol2d)  # smaller than current → unchanged
    assert out.image.shape == vol2d.image.shape


def test_pad_constant_fill_value(vol2d):
    out = Pad(size=(50, 70), mode="constant", cval=7.0)(vol2d)
    assert out.image[0, 0] == pytest.approx(7.0)


def test_pad_mask_filled_with_zero(vol3d):
    out = Pad(size=(16, 40, 40))(vol3d)
    assert out.mask[0, 0, 0] == 0
    assert out.mask.dtype == vol3d.mask.dtype


def test_pad_to_dict_round_trip():
    t = Pad(size=(64, 64), mode="edge", cval=0.0, p=1.0)
    t2 = Pad(**t.to_dict()["params"])
    assert t2.size == t.size
    assert t2.mode == t.mode


# ---- CenterCrop ----

def test_center_crop_to_target(vol2d):
    out = CenterCrop(size=(20, 20))(vol2d)
    assert out.image.shape == (20, 20)
    assert out.mask.shape == (20, 20)


def test_center_crop_is_centred(vol2d):
    out = CenterCrop(size=(20, 20))(vol2d)
    expected = vol2d.image[10:30, 20:40]
    np.testing.assert_array_equal(out.image, expected)


def test_center_crop_never_pads(vol2d):
    out = CenterCrop(size=(100, 100))(vol2d)
    assert out.image.shape == vol2d.image.shape


def test_pad_then_center_crop_forces_exact_shape(vol2d):
    target = (50, 50)
    out = CenterCrop(size=target)(Pad(size=target)(vol2d))
    assert out.image.shape == target


def test_center_crop_to_dict_round_trip():
    t = CenterCrop(size=(16, 16))
    t2 = CenterCrop(**t.to_dict()["params"])
    assert t2.size == t.size
