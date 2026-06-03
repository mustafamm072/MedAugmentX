import numpy as np
import pytest

from medaugmentx import MedVolume
from medaugmentx.interop import MonaiMapTransform, SampleTransform, TorchTransform
from medaugmentx.transforms import RandomFlip


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
