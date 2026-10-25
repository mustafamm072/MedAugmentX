import copy
import json

import numpy as np
import pytest

from medaugmentx import Compose, Guard, MedVolume, OneOf
from medaugmentx.reproducibility import (
    apply_for_sample,
    record_augmentation,
    replay_augmentation,
    seed_for_sample,
)
from medaugmentx.serialization import from_dict, from_json, from_yaml, to_json
from medaugmentx.transforms import AnisotropicElastic, GaussianNoise, RandomFlip


def test_per_sample_seed_is_stable_and_distinguishes_ids_and_epochs():
    assert seed_for_sample(42, "case1", epoch=1) == seed_for_sample(42, "case1", epoch=1)
    assert (
        len(
            {
                seed_for_sample(42, key, epoch=epoch)
                for key in ("case1", "case2")
                for epoch in (1, 2)
            }
        )
        == 4
    )
    assert seed_for_sample(42, 1) != seed_for_sample(42, "1")


@pytest.mark.parametrize("seed", [-1, 2.5, True])
def test_seed_validation(seed):
    with pytest.raises(ValueError):
        seed_for_sample(seed, "x")


def test_per_sample_does_not_advance_original():
    volume = MedVolume(np.zeros((8, 8), np.float32))
    pipeline = Compose([GaussianNoise(std=0.1)], seed=7)
    untouched = copy.deepcopy(pipeline)
    a = apply_for_sample(pipeline, volume, seed=1, sample_id="a")
    b = apply_for_sample(pipeline, volume, seed=1, sample_id="a")
    np.testing.assert_array_equal(a.image, b.image)
    np.testing.assert_array_equal(pipeline(volume).image, untouched(volume).image)


@pytest.mark.parametrize("bit_generator", [np.random.PCG64, np.random.MT19937, np.random.Philox])
def test_execution_record_replays_an_advanced_nested_pipeline(bit_generator):
    pipeline = Guard(
        Compose(
            [
                OneOf([RandomFlip(p_per_axis=1), GaussianNoise(std=0.1)]),
                AnisotropicElastic(alpha=(2, 3, 3), p=0.7),
            ]
        ),
        seed=7,
    )
    pipeline.reseed(np.random.Generator(bit_generator(12)))
    volume = MedVolume(np.random.default_rng(4).random((4, 8, 8)).astype(np.float32))
    pipeline(volume)  # constructor seeds cannot replay this next draw
    output, record = record_augmentation(pipeline, volume)
    record = json.loads(json.dumps(record))
    replay = replay_augmentation(record, volume)
    np.testing.assert_array_equal(output.image, replay.image)
    assert record["trace"]
    assert all(set(event) == {"path", "name", "action"} for event in record["trace"])
    modified = volume.replace(image=volume.image + 1)
    with pytest.raises(ValueError, match="input fingerprint"):
        replay_augmentation(record, modified)


def test_versioned_policy_rejects_tampering_and_future_schemas():
    policy = json.loads(to_json(GaussianNoise(seed=1)))
    assert policy["schema_version"] == 1
    policy["pipeline"]["params"]["std"] = 999
    with pytest.raises(ValueError, match="hash mismatch"):
        from_dict(policy)
    policy["schema_version"] = 2
    with pytest.raises(ValueError, match="schema version"):
        from_dict(policy)


@pytest.mark.parametrize(
    "loader,document",
    [
        (from_json, '{"name":"GaussianNoise","params":{"std":1e309}}'),
        (from_yaml, "name: GaussianNoise\nparams: {std: .inf}"),
        (from_dict, {"name": "GaussianNoise", "params": {"std": np.nan}}),
    ],
)
def test_nonfinite_policy_values_rejected_consistently(loader, document):
    with pytest.raises(ValueError, match="finite"):
        loader(document)


def test_transform_ownership_prevents_rng_aliasing():
    transform = GaussianNoise()
    with pytest.raises(ValueError, match="shared"):
        Compose([transform, transform])
    first = Compose([transform])
    with pytest.raises(ValueError, match="shared"):
        OneOf([transform])
    assert len(first) == 1
