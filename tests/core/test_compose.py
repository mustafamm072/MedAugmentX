import numpy as np
import pytest

from medaugmentx.core import Compose, MedVolume, OneOf, SomeOf, Transform


class AddConstant(Transform):
    def __init__(self, value, p=1.0, seed=None):
        super().__init__(p=p, seed=seed)
        self.value = value

    def apply(self, volume):
        return volume.replace(image=volume.image + self.value)


class MultiplyConstant(Transform):
    def __init__(self, value, p=1.0, seed=None):
        super().__init__(p=p, seed=seed)
        self.value = value

    def apply(self, volume):
        return volume.replace(image=volume.image * self.value)


@pytest.fixture
def vol():
    return MedVolume(image=np.zeros((4, 4), dtype=np.float32))


class TestCompose:
    def test_sequential_application(self, vol):
        out = Compose([AddConstant(2.0), MultiplyConstant(3.0)])(vol)
        assert np.all(out.image == 6.0)

    def test_empty_compose_is_identity(self, vol):
        out = Compose([])(vol)
        np.testing.assert_array_equal(out.image, vol.image)

    def test_seed_makes_output_reproducible(self, vol):
        a = Compose([AddConstant(1.0)], seed=42)(vol)
        b = Compose([AddConstant(1.0)], seed=42)(vol)
        np.testing.assert_array_equal(a.image, b.image)

    def test_rejects_non_transform_children(self):
        with pytest.raises(TypeError, match="Compose expected"):
            Compose([lambda v: v])  # type: ignore[list-item]

    def test_rejects_non_volume_input(self):
        with pytest.raises(TypeError, match="MedVolume"):
            Compose([])(42)  # type: ignore[arg-type]


class TestOneOf:
    def test_one_branch_runs(self, vol):
        out = OneOf([AddConstant(1.0), AddConstant(7.0)], seed=0)(vol)
        # exactly one of the constants is added
        assert np.all((out.image == 1.0) | (out.image == 7.0))

    def test_seed_determines_branch(self, vol):
        a = OneOf([AddConstant(1.0), AddConstant(7.0)], seed=42)(vol)
        b = OneOf([AddConstant(1.0), AddConstant(7.0)], seed=42)(vol)
        np.testing.assert_array_equal(a.image, b.image)

    def test_weights_must_be_positive(self):
        with pytest.raises(ValueError):
            OneOf([AddConstant(1.0), AddConstant(2.0)], weights=[-1, 1])

    def test_empty_oneof_rejected(self):
        with pytest.raises(ValueError):
            OneOf([])

    def test_p_zero_returns_identity(self, vol):
        out = OneOf([AddConstant(99.0)], p=0.0, seed=0)(vol)
        np.testing.assert_array_equal(out.image, vol.image)


class TestSomeOf:
    def test_picks_n_children(self, vol):
        out = SomeOf([AddConstant(1.0), AddConstant(10.0), AddConstant(100.0)], n=2, seed=0)(vol)
        # Sum of two distinct constants from {1, 10, 100}
        possible = {1 + 10, 1 + 100, 10 + 100}
        assert float(out.image[0, 0]) in possible

    def test_n_zero_is_identity(self, vol):
        out = SomeOf([AddConstant(1.0), AddConstant(2.0)], n=0)(vol)
        np.testing.assert_array_equal(out.image, vol.image)

    def test_n_too_large_rejected(self):
        with pytest.raises(ValueError):
            SomeOf([AddConstant(1.0)], n=2)

    def test_range_n_works(self, vol):
        out = SomeOf([AddConstant(1.0), AddConstant(2.0)], n=(0, 2), seed=0)(vol)
        # value is sum of a subset of {1, 2}
        assert float(out.image[0, 0]) in {0.0, 1.0, 2.0, 3.0}
