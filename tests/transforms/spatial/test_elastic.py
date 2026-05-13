import numpy as np

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
