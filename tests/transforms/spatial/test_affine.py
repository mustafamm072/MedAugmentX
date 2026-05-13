import numpy as np

from medaugmentx.core import MedVolume
from medaugmentx.transforms import RandomAffine


def test_zero_params_is_near_identity():
    img = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    vol = MedVolume(image=img)
    out = RandomAffine(rotation=0.0, scale=(1.0, 1.0), translation=(0.0, 0.0), seed=0)(vol)
    np.testing.assert_allclose(out.image, img, atol=1e-5)


def test_2d_rotation_changes_image():
    img = np.zeros((32, 32), dtype=np.float32)
    img[10:22, 14:18] = 1.0  # vertical bar
    vol = MedVolume(image=img)
    out = RandomAffine(rotation=(45.0, 45.0), seed=0)(vol)
    # rotation by 45 degrees is not identity
    assert not np.allclose(out.image, img)
    # output preserves shape
    assert out.image.shape == img.shape


def test_3d_axes_disabled_only_rotates_enabled():
    img = np.zeros((8, 16, 16), dtype=np.float32)
    img[3:5, 6:10, 6:10] = 1.0
    vol = MedVolume(image=img)
    out = RandomAffine(
        rotation=(15.0, 15.0),
        axes_enabled=("x", "y"),  # disable z rotation
        seed=0,
    )(vol)
    assert out.image.shape == img.shape


def test_mask_uses_nearest_neighbour_preserving_label_set():
    img = np.zeros((32, 32), dtype=np.float32)
    img[10:22, 10:22] = 1.0
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[12:20, 12:20] = 3
    vol = MedVolume(image=img, mask=mask)
    out = RandomAffine(rotation=15.0, seed=7)(vol)
    # mask should still be uint8 and contain only labels {0, 3}
    assert out.mask.dtype == np.uint8
    assert set(np.unique(out.mask).tolist()).issubset({0, 3})


def test_seed_reproducible():
    img = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    vol = MedVolume(image=img)
    a = RandomAffine(rotation=10.0, scale=(0.9, 1.1), seed=42)(vol)
    b = RandomAffine(rotation=10.0, scale=(0.9, 1.1), seed=42)(vol)
    np.testing.assert_array_equal(a.image, b.image)


def test_p_zero_is_identity():
    img = np.arange(64, dtype=np.float32).reshape(8, 8)
    vol = MedVolume(image=img)
    out = RandomAffine(rotation=45.0, p=0.0, seed=0)(vol)
    np.testing.assert_array_equal(out.image, img)
