import copy
import json

import numpy as np
import pytest

from medaugmentx import (
    MedVolume,
    list_model_cards,
    model_card,
    preview_augmentation,
    volume_summary,
)
from medaugmentx import transforms as T
from medaugmentx.serialization import from_json, to_json


def test_every_transform_has_packaged_model_card():
    assert set(list_model_cards()) == set(T.__all__)
    for name in T.__all__:
        card = model_card(name)
        assert card["clinical_validation"] is False
        assert card["assumptions"] and card["limitations"]
    card = model_card("RicianNoise")
    card["assumptions"] = "mutated"
    assert "mutated" not in model_card("RicianNoise")["assumptions"]


def test_noise_moments_against_gaussian_and_rayleigh_oracles():
    sigma = 0.2
    volume = MedVolume(np.zeros((700, 700), np.float32))
    gaussian = T.GaussianNoise(std=sigma, seed=42)(volume).image
    assert abs(gaussian.mean()) < 0.001
    assert gaussian.var() == pytest.approx(sigma**2, rel=0.01)
    magnitude = T.RicianNoise(std=sigma, seed=42)(volume).image
    assert magnitude.mean() == pytest.approx(sigma * np.sqrt(np.pi / 2), rel=0.005)
    assert magnitude.var() == pytest.approx((2 - np.pi / 2) * sigma**2, rel=0.01)


def test_blur_frequency_response_and_bias_bound():
    x = np.arange(256)
    image = np.broadcast_to(1 + 0.2 * np.cos(2 * np.pi * x / 16), (32, 256)).copy()
    blurred = T.GaussianBlur(sigma=2)(MedVolume(image)).image
    # Interior analytic response of a Gaussian on a sinusoid.
    expected = 1 + 0.2 * np.exp(-0.5 * (2 * np.pi * 2 / 16) ** 2) * np.cos(2 * np.pi * x / 16)
    np.testing.assert_allclose(
        blurred[:, 16:-16], np.broadcast_to(expected[16:-16], (32, 224)), atol=1e-5
    )
    biased = T.BiasField(alpha=0.4, seed=3)(MedVolume(np.ones((48, 48)))).image
    assert biased.min() >= np.exp(-0.4) - 1e-6
    assert biased.max() <= np.exp(0.4) + 1e-6


def test_normalization_and_antialiasing():
    normalized = T.NormalizeIntensity()(MedVolume(np.arange(1024).reshape(32, 32)))
    assert normalized.image.min() == 0 and normalized.image.max() == 1
    assert not T.NormalizeIntensity()(MedVolume(np.ones((3, 3)))).image.any()
    yy, xx = np.mgrid[:64, :64]
    high_frequency = MedVolume(np.cos(xx * 2 * np.pi / 3) + np.cos(yy * 2 * np.pi / 3))
    filtered = T.Resize((16, 16), antialias=True)(high_frequency).image
    aliased = T.Resize((16, 16), antialias=False)(high_frequency).image
    assert filtered.var() < aliased.var() * 0.1


def test_laterality_remapping_round_trip_and_involution():
    volume = MedVolume(
        np.arange(12).reshape(3, 4),
        mask=np.array([[1, 1, 2, 2]] * 3),
        keypoints=[[1, 0], [1, 3]],
        keypoint_labels=["left", "right"],
    )
    flip = T.RandomFlip(
        axes=("x",),
        p_per_axis=1,
        label_map={1: 2, 2: 1},
        keypoint_label_map={"left": "right", "right": "left"},
    )
    restored = from_json(to_json(flip))
    out = restored(restored(volume))
    np.testing.assert_array_equal(out.image, volume.image)
    np.testing.assert_array_equal(out.mask, volume.mask)
    np.testing.assert_array_equal(out.keypoint_labels, volume.keypoint_labels)
    with pytest.raises(ValueError, match="involut"):
        T.RandomFlip(label_map={1: 2})


def test_preview_isolated_policy_and_serializable_report(tmp_path):
    pytest.importorskip("matplotlib")
    policy = T.GaussianNoise(seed=1)
    expected = copy.deepcopy(policy)(MedVolume(np.arange(64).reshape(8, 8))).image
    volume = MedVolume(np.arange(64).reshape(8, 8), mask=np.eye(8, dtype=np.uint8))
    path = tmp_path / "preview.png"
    report = preview_augmentation(policy, volume, str(path), samples=2)
    assert path.stat().st_size > 1000
    assert len(report["samples"]) == 3
    json.dumps(report, allow_nan=False)
    np.testing.assert_array_equal(policy(volume).image, expected)
    assert volume_summary(volume)["labels"][1]["voxels"] == 8
