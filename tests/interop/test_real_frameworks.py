"""Actual optional framework objects, independent affine oracles, and worker replay."""

from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from medaugmentx import Compose, MedVolume
from medaugmentx.interop import MonaiMapTransform, TorchIOTransform, TorchTransform
from medaugmentx.reproducibility import apply_for_sample
from medaugmentx.transforms import CenterCrop, GaussianNoise, RandomFlip, Resize

pytestmark = pytest.mark.frameworks


def _affine():
    return np.array([[1, 0.2, 0, 10], [0, 2, 0, -4], [0, 0, 3, 7], [0, 0, 0, 1]], float)


def test_torchio_native_x_flip_and_affine_crop_resize():
    tio = pytest.importorskip("torchio")
    data = torch.arange(7 * 9 * 11, dtype=torch.float32).reshape(1, 7, 9, 11)
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=data, affine=_affine()),
        label=tio.LabelMap(tensor=(data > 200).to(torch.uint8), affine=_affine()),
    )
    flip = TorchIOTransform(RandomFlip(axes=("x",), p_per_axis=1))
    np.testing.assert_array_equal(flip(subject).image.data.numpy(), data.flip(1).numpy())
    policy = Compose([CenterCrop((7, 5, 3)), Resize((9, 7, 5))])
    out = TorchIOTransform(policy)(subject)
    expected = _affine().copy()
    # Native crop origins: (2,2,2). Voxel-extent resize pull: (.6,5/7,7/9).
    pull = np.diag([3 / 5, 5 / 7, 7 / 9, 1.0])
    pull[:3, 3] = np.array([2, 2, 2]) + (np.diag(pull)[:3] - 1) / 2
    np.testing.assert_allclose(out.image.affine, expected @ pull)
    np.testing.assert_allclose(out.label.affine, out.image.affine)
    assert out.image.shape == out.label.shape == (1, 5, 7, 9)
    np.testing.assert_array_equal(subject.image.data.numpy(), data.numpy())
    np.testing.assert_allclose(subject.image.affine, _affine())


def test_torchio_rejects_wrong_label_grid_and_missing_keys():
    tio = pytest.importorskip("torchio")
    data = torch.zeros((1, 5, 7, 9))
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=data, affine=_affine()),
        label=tio.LabelMap(tensor=data, affine=np.eye(4)),
    )
    with pytest.raises(ValueError, match="grids disagree"):
        TorchIOTransform(RandomFlip())(subject)
    with pytest.raises(KeyError, match="label_key"):
        TorchIOTransform(RandomFlip(), label_key="missing")(subject)
    subject["other"] = tio.ScalarImage(tensor=data)
    with pytest.raises(ValueError, match="Multiple images"):
        TorchIOTransform(RandomFlip(), image_key="image")(subject)


def test_monai_meta_tensor_geometry_and_history_are_explicit():
    monai = pytest.importorskip("monai")
    data = torch.arange(7 * 9 * 11, dtype=torch.float32).reshape(1, 7, 9, 11)
    image = monai.data.MetaTensor(data, affine=_affine(), applied_operations=[{"class": "old"}])
    label = monai.data.MetaTensor((data > 200).to(torch.int16), affine=_affine())
    adapter = MonaiMapTransform(CenterCrop((7, 5, 3)))
    out = adapter({"image": image, "label": label})
    expected = _affine().copy()
    expected[:3, 3] += expected[:3, :3] @ [2, 2, 2]
    np.testing.assert_allclose(out["image"].affine, expected)
    np.testing.assert_allclose(out["label"].affine, expected)
    assert isinstance(out["image"], monai.data.MetaTensor)
    assert out["label"].dtype == torch.int16
    assert out["image"].applied_operations == []
    assert out["image"].meta["medaugmentx_noninvertible"]
    assert image.applied_operations == [{"class": "old"}]
    np.testing.assert_allclose(image.affine, _affine())


def test_torch_mapping_geometry_and_targets():
    data = torch.arange(8 * 10, dtype=torch.float32).reshape(1, 8, 10)
    sample = {
        "image": data,
        "mask": torch.ones_like(data, dtype=torch.uint8),
        "affine": np.eye(4),
        "keypoints": torch.tensor([[4.0, 5.0]]),
        "keypoint_labels": torch.tensor([42]),
    }
    out = TorchTransform(CenterCrop((4, 6)), channel_dim=0)(sample)
    np.testing.assert_allclose(out["keypoints"], [[2, 3]])
    np.testing.assert_allclose(out["affine"][:3, 3], [2, 2, 0])
    assert out["image"].device == data.device
    assert out["mask"].dtype == torch.uint8


class _SeededDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.policy = Compose([GaussianNoise(std=0.1)], seed=7)

    def __len__(self):
        return 8

    def __getitem__(self, index):
        out = apply_for_sample(
            self.policy, MedVolume(np.zeros((4, 4), np.float32)), seed=123, sample_id=index, epoch=2
        )
        return torch.from_numpy(out.image)


def test_sample_outputs_are_independent_of_worker_assignment():
    dataset = _SeededDataset()
    single = torch.cat(list(torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)))
    multi = torch.cat(
        list(
            torch.utils.data.DataLoader(
                copy.deepcopy(dataset), batch_size=2, num_workers=2, multiprocessing_context="spawn"
            )
        )
    )
    torch.testing.assert_close(single, multi, rtol=0, atol=0)
    assert not torch.equal(single[0], single[1])
