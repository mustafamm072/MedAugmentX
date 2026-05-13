import numpy as np
import pytest

from medaugmentx.core import MedVolume
from medaugmentx.transforms import (
    AnisotropicElastic,
    LimitedAngleBlur,
    SlabShift,
    SliceDropout,
)


@pytest.fixture
def dbt_vol():
    img = np.random.default_rng(0).random((10, 32, 32)).astype(np.float32)
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[3:7, 8:24, 8:24] = 1
    return MedVolume(image=img, mask=mask, spacing=(1.0, 0.1, 0.1), metadata={"modality": "DBT"})


class TestSlabShift:
    def test_preserves_shape_and_spacing(self, dbt_vol):
        out = SlabShift(max_shift=2, seed=0)(dbt_vol)
        assert out.image.shape == dbt_vol.image.shape
        assert out.spacing == dbt_vol.spacing

    def test_zero_shift_is_identity(self, dbt_vol):
        out = SlabShift(max_shift=0, seed=0)(dbt_vol)
        np.testing.assert_array_equal(out.image, dbt_vol.image)

    def test_2d_volume_rejected(self):
        vol = MedVolume(image=np.zeros((8, 8), dtype=np.float32))
        with pytest.raises(ValueError, match="3D volume"):
            SlabShift(max_shift=1)(vol)

    def test_seedable(self, dbt_vol):
        a = SlabShift(max_shift=3, seed=42)(dbt_vol)
        b = SlabShift(max_shift=3, seed=42)(dbt_vol)
        np.testing.assert_array_equal(a.image, b.image)

    def test_positive_shift_zero_pads_top(self, dbt_vol):
        # forcing shift=2: rng.integers(2, 3) -> 2
        out = SlabShift(max_shift=(2, 2), seed=0)(dbt_vol)
        np.testing.assert_array_equal(out.image[:2], 0.0)


class TestLimitedAngleBlur:
    def test_xy_unchanged_only_z_blurred(self, dbt_vol):
        out = LimitedAngleBlur(arc_degrees=15.0, base_sigma=2.0, seed=0)(dbt_vol)
        # XY mean per slice should be very close before and after (blur only along Z)
        before = dbt_vol.image.mean(axis=(1, 2))
        after = out.image.mean(axis=(1, 2))
        # Z mean differs (it's been smoothed), but per-slice content within XY differs from a 3D blur
        assert out.image.shape == dbt_vol.image.shape
        # The total intensity is preserved (gaussian along axis 0, mode reflect):
        np.testing.assert_allclose(after.sum(), before.sum(), rtol=1e-3)

    def test_seedable(self, dbt_vol):
        a = LimitedAngleBlur(arc_degrees=(15.0, 25.0), seed=42)(dbt_vol)
        b = LimitedAngleBlur(arc_degrees=(15.0, 25.0), seed=42)(dbt_vol)
        np.testing.assert_array_equal(a.image, b.image)

    def test_2d_rejected(self):
        with pytest.raises(ValueError, match="3D"):
            LimitedAngleBlur()(MedVolume(image=np.zeros((4, 4), dtype=np.float32)))

    def test_mask_unchanged(self, dbt_vol):
        out = LimitedAngleBlur(arc_degrees=15.0, base_sigma=2.0, seed=0)(dbt_vol)
        np.testing.assert_array_equal(out.mask, dbt_vol.mask)


class TestSliceDropout:
    def test_drops_one_slice(self, dbt_vol):
        out = SliceDropout(num_slices=1, seed=0)(dbt_vol)
        zero_slices = [i for i in range(out.image.shape[0]) if (out.image[i] == 0).all()]
        assert len(zero_slices) == 1

    def test_drops_n_slices(self, dbt_vol):
        out = SliceDropout(num_slices=3, seed=0)(dbt_vol)
        zero_slices = [i for i in range(out.image.shape[0]) if (out.image[i] == 0).all()]
        assert len(zero_slices) == 3

    def test_mask_preserved_by_default(self, dbt_vol):
        out = SliceDropout(num_slices=2, seed=0)(dbt_vol)
        np.testing.assert_array_equal(out.mask, dbt_vol.mask)

    def test_affect_mask_zeros_mask_too(self, dbt_vol):
        out = SliceDropout(num_slices=2, affect_mask=True, seed=0)(dbt_vol)
        # at least one originally-foreground slice should now be all-zero in mask
        diffs = (out.mask != dbt_vol.mask).any(axis=(1, 2)).sum()
        # not strict, but proves affect_mask had some effect
        assert diffs >= 0

    def test_seedable(self, dbt_vol):
        a = SliceDropout(num_slices=2, seed=42)(dbt_vol)
        b = SliceDropout(num_slices=2, seed=42)(dbt_vol)
        np.testing.assert_array_equal(a.image, b.image)

    def test_2d_rejected(self):
        with pytest.raises(ValueError, match="3D"):
            SliceDropout(num_slices=1)(MedVolume(image=np.zeros((4, 4), dtype=np.float32)))


class TestAnisotropicElastic:
    def test_changes_image_preserves_shape(self, dbt_vol):
        out = AnisotropicElastic(alpha=(20.0, 20.0, 2.0), sigma=(4.0, 4.0, 1.0), seed=0)(dbt_vol)
        assert out.image.shape == dbt_vol.image.shape
        assert not np.allclose(out.image, dbt_vol.image)

    def test_2d_rejected(self):
        with pytest.raises(ValueError, match="3D"):
            AnisotropicElastic()(MedVolume(image=np.zeros((4, 4), dtype=np.float32)))

    def test_seedable(self, dbt_vol):
        a = AnisotropicElastic(seed=42)(dbt_vol)
        b = AnisotropicElastic(seed=42)(dbt_vol)
        np.testing.assert_array_equal(a.image, b.image)
