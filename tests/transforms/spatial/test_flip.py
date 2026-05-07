import numpy as np

from medaugment.core import MedVolume
from medaugment.transforms import RandomFlip


def test_flip_x_2d_changes_image_and_mask_consistently():
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    mask = (img > 8).astype(np.uint8)
    vol = MedVolume(image=img, mask=mask)
    out = RandomFlip(axes=("x",), p_per_axis=1.0, seed=0)(vol)
    np.testing.assert_array_equal(out.image, np.flip(img, axis=1))
    np.testing.assert_array_equal(out.mask, np.flip(mask, axis=1))


def test_flip_x_3d_uses_axis_2():
    img = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    vol = MedVolume(image=img, mask=img.astype(np.uint8))
    out = RandomFlip(axes=("x",), p_per_axis=1.0, seed=0)(vol)
    np.testing.assert_array_equal(out.image, np.flip(img, axis=2))


def test_flip_z_3d_uses_axis_0():
    img = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    vol = MedVolume(image=img)
    out = RandomFlip(axes=("z",), p_per_axis=1.0, seed=0)(vol)
    np.testing.assert_array_equal(out.image, np.flip(img, axis=0))


def test_flip_p_per_axis_zero_is_identity():
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    vol = MedVolume(image=img)
    out = RandomFlip(axes=("x",), p_per_axis=0.0, seed=0)(vol)
    np.testing.assert_array_equal(out.image, img)


def test_flip_is_seedable():
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    vol = MedVolume(image=img)
    a = RandomFlip(axes=("x", "y"), p_per_axis=0.5, seed=42)(vol)
    b = RandomFlip(axes=("x", "y"), p_per_axis=0.5, seed=42)(vol)
    np.testing.assert_array_equal(a.image, b.image)


def test_flip_preserves_spacing_and_metadata():
    vol = MedVolume(
        image=np.zeros((4, 4), dtype=np.float32),
        spacing=(0.5, 0.5),
        metadata={"modality": "DX", "patient_id": "p1"},
    )
    out = RandomFlip(axes=("x",), p_per_axis=1.0, seed=0)(vol)
    assert out.spacing == (0.5, 0.5)
    assert out.metadata == {"modality": "DX", "patient_id": "p1"}
