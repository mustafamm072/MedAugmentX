import numpy as np
import pytest

from medaugmentx import MedVolume
from medaugmentx.interop import (
    MonaiMapTransform,
    SampleTransform,
    TorchIOTransform,
    TorchTransform,
)
from medaugmentx.transforms import RandomFlip


class FakeScalarImage:
    def __init__(self, data, spacing=(1.0, 1.0), affine=None):
        self.data = data
        self.spacing = spacing
        self.affine = affine

    def set_data(self, data):
        self.data = data

    def copy(self):
        return self.__class__(self.data.copy(), spacing=self.spacing, affine=self.affine)


class CopyReturnsDictImage(FakeScalarImage):
    def copy(self):
        return {"data": self.data.copy(), "spacing": self.spacing}


class FakeLabelMap(FakeScalarImage):
    pass


class FakeSubject(dict):
    def copy(self):
        return FakeSubject(
            {
                key: value.copy() if hasattr(value, "copy") else value
                for key, value in self.items()
            }
        )


def test_sample_transform_accepts_medvolume(vol2d):
    adapter = SampleTransform(RandomFlip(axes=("x",), p_per_axis=1.0, seed=0))

    out = adapter(vol2d)

    assert isinstance(out, MedVolume)
    np.testing.assert_array_equal(out.image, np.flip(vol2d.image, axis=1))
    np.testing.assert_array_equal(out.mask, np.flip(vol2d.mask, axis=1))


def test_torch_transform_accepts_mapping_samples():
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    mask = np.arange(6, dtype=np.uint8).reshape(2, 3)
    sample = {
        "image": image,
        "mask": mask,
        "spacing": (0.8, 0.8),
        "metadata": {"modality": "DX"},
        "case_id": "case-001",
    }
    adapter = TorchTransform(RandomFlip(axes=("x",), p_per_axis=1.0, seed=0))

    out = adapter(sample)

    assert out is not sample
    assert out["case_id"] == "case-001"
    np.testing.assert_array_equal(out["image"], np.flip(image, axis=1))
    np.testing.assert_array_equal(out["mask"], np.flip(mask, axis=1))
    assert out["mask"].dtype == mask.dtype
    np.testing.assert_array_equal(sample["image"], image)


def test_torch_transform_restores_singleton_channel_dimension():
    image = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    mask = np.arange(6, dtype=np.uint8).reshape(1, 2, 3)
    adapter = TorchTransform(RandomFlip(axes=("x",), p_per_axis=1.0, seed=0))

    out_image, out_mask = adapter((image, mask))

    assert out_image.shape == image.shape
    assert out_mask.shape == mask.shape
    np.testing.assert_array_equal(out_image, np.flip(image, axis=2))
    np.testing.assert_array_equal(out_mask, np.flip(mask, axis=2))


def test_torch_transform_preserves_sequence_type_for_lists():
    image = np.arange(4, dtype=np.float32).reshape(2, 2)
    mask = np.ones((2, 2), dtype=np.uint8)
    adapter = TorchTransform(RandomFlip(axes=("x",), p_per_axis=1.0, seed=0))

    out = adapter([image, mask])

    assert isinstance(out, list)
    np.testing.assert_array_equal(out[0], np.flip(image, axis=1))


def test_monai_map_transform_uses_label_key_by_default():
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    label = np.arange(6, dtype=np.uint8).reshape(2, 3)
    adapter = MonaiMapTransform(RandomFlip(axes=("x",), p_per_axis=1.0, seed=0))

    out = adapter({"image": image, "label": label})

    np.testing.assert_array_equal(out["image"], np.flip(image, axis=1))
    np.testing.assert_array_equal(out["label"], np.flip(label, axis=1))


def test_channel_dim_must_be_singleton():
    image = np.zeros((2, 4, 4), dtype=np.float32)
    adapter = TorchTransform(
        RandomFlip(axes=("x",), p_per_axis=1.0, seed=0),
        channel_dim=0,
    )

    with pytest.raises(ValueError, match="single-channel"):
        adapter(image)


def test_mapping_requires_image_key():
    adapter = TorchTransform(RandomFlip(axes=("x",), p_per_axis=1.0, seed=0))

    with pytest.raises(KeyError, match="image_key"):
        adapter({"mask": np.zeros((2, 2), dtype=np.uint8)})


def test_mapping_metadata_must_be_mapping():
    adapter = TorchTransform(RandomFlip(axes=("x",), p_per_axis=1.0, seed=0))

    with pytest.raises(TypeError, match="metadata_key"):
        adapter({"image": np.zeros((2, 2), dtype=np.float32), "metadata": "bad"})


def test_torchio_transform_updates_subject_copy_and_preserves_original():
    image = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    label = np.arange(6, dtype=np.uint8).reshape(1, 2, 3)
    subject = FakeSubject(
        {
            "image": FakeScalarImage(image, spacing=(0.8, 0.8)),
            "label": FakeLabelMap(label, spacing=(0.8, 0.8)),
            "case_id": "case-001",
        }
    )
    adapter = TorchIOTransform(RandomFlip(axes=("x",), p_per_axis=1.0, seed=0))

    out = adapter(subject)

    assert isinstance(out, FakeSubject)
    assert out is not subject
    assert out["case_id"] == "case-001"
    np.testing.assert_array_equal(out["image"].data, np.flip(image, axis=2))
    np.testing.assert_array_equal(out["label"].data, np.flip(label, axis=2))
    np.testing.assert_array_equal(subject["image"].data, image)
    np.testing.assert_array_equal(subject["label"].data, label)


def test_torchio_transform_infers_scalar_and_label_keys():
    image = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    label = np.arange(6, dtype=np.uint8).reshape(1, 2, 3)
    subject = FakeSubject({"t1": FakeScalarImage(image), "seg": FakeLabelMap(label)})
    adapter = TorchIOTransform(RandomFlip(axes=("x",), p_per_axis=1.0, seed=0))

    out = adapter(subject)

    np.testing.assert_array_equal(out["t1"].data, np.flip(image, axis=2))
    np.testing.assert_array_equal(out["seg"].data, np.flip(label, axis=2))


def test_torchio_transform_accepts_single_image_object():
    image = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    adapter = TorchIOTransform(RandomFlip(axes=("x",), p_per_axis=1.0, seed=0))

    out = adapter(FakeScalarImage(image))

    assert isinstance(out, FakeScalarImage)
    np.testing.assert_array_equal(out.data, np.flip(image, axis=2))


def test_torchio_transform_does_not_use_image_copy_method_for_objects():
    image = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    subject = FakeSubject({"image": CopyReturnsDictImage(image)})
    adapter = TorchIOTransform(RandomFlip(axes=("x",), p_per_axis=1.0, seed=0))

    out = adapter(subject)

    assert isinstance(out["image"], CopyReturnsDictImage)
    np.testing.assert_array_equal(out["image"].data, np.flip(image, axis=2))
    np.testing.assert_array_equal(subject["image"].data, image)


def test_torchio_transform_falls_back_to_generic_array_samples():
    image = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    adapter = TorchIOTransform(RandomFlip(axes=("x",), p_per_axis=1.0, seed=0))

    out = adapter(image)

    np.testing.assert_array_equal(out, np.flip(image, axis=2))
