import numpy as np
import pytest

from medaugmentx.core import MedVolume


class TestMedVolume:
    def test_2d_construction(self):
        img = np.zeros((10, 10), dtype=np.float32)
        v = MedVolume(image=img)
        assert v.ndim == 2
        assert v.spacing == (1.0, 1.0)
        assert v.mask is None
        assert v.metadata == {}

    def test_3d_construction_with_mask(self):
        img = np.zeros((4, 8, 8), dtype=np.float32)
        mask = np.zeros_like(img, dtype=np.uint8)
        v = MedVolume(image=img, mask=mask, spacing=(1.0, 0.5, 0.5), metadata={"m": 1})
        assert v.is_3d
        assert v.shape == (4, 8, 8)
        assert v.has_mask
        assert v.spacing == (1.0, 0.5, 0.5)
        assert v.metadata == {"m": 1}

    def test_rejects_4d(self):
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            MedVolume(image=np.zeros((1, 2, 3, 4), dtype=np.float32))

    def test_rejects_mask_shape_mismatch(self):
        img = np.zeros((4, 4), dtype=np.float32)
        mask = np.zeros((5, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="mask shape"):
            MedVolume(image=img, mask=mask)

    def test_rejects_spacing_ndim_mismatch(self):
        img = np.zeros((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="spacing has"):
            MedVolume(image=img, spacing=(1.0, 1.0, 1.0))

    def test_rejects_non_array_image(self):
        with pytest.raises(TypeError, match="image must be a numpy.ndarray"):
            MedVolume(image=[[1, 2], [3, 4]])  # type: ignore[arg-type]

    def test_replace_swaps_image(self):
        img = np.zeros((4, 4), dtype=np.float32)
        v = MedVolume(image=img, spacing=(1.0, 1.0))
        v2 = v.replace(image=np.ones((4, 4), dtype=np.float32))
        assert np.all(v2.image == 1.0)
        assert np.all(v.image == 0.0)
        assert v2.spacing == (1.0, 1.0)

    def test_replace_metadata_is_shallow_copy(self):
        v = MedVolume(image=np.zeros((4, 4), dtype=np.float32), metadata={"a": 1})
        v2 = v.replace()
        v2.metadata["a"] = 99
        assert v.metadata["a"] == 1

    def test_copy_deep_copies(self):
        img = np.ones((4, 4), dtype=np.float32)
        v = MedVolume(image=img, mask=img.astype(np.uint8))
        c = v.copy()
        c.image[0, 0] = 42
        assert v.image[0, 0] == 1.0

    def test_modality_property(self):
        v = MedVolume(image=np.zeros((4, 4), dtype=np.float32), metadata={"modality": "MR"})
        assert v.modality == "MR"

    def test_repr_smoke(self):
        v = MedVolume(image=np.zeros((4, 4), dtype=np.float32))
        s = repr(v)
        assert "MedVolume" in s and "shape" in s
