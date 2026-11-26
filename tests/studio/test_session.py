"""StudioSession: data loading, policy editing, previews and exports."""

import json
import os

import numpy as np
import pytest

from medaugmentx import (
    Compose,
    Guard,
    MedVolume,
    PerChannel,
    apply_for_sample,
    replay_augmentation,
)
from medaugmentx.serialization import _make_serialisable, from_dict, from_json, from_yaml
from medaugmentx.studio import PolicyError, StudioSession, python_snippet
from medaugmentx.transforms import GaussianNoise, RandomAffine, RandomFlip


@pytest.fixture
def session():
    studio = StudioSession()
    studio.load_phantom("mri")
    yield studio
    studio.close()


def noise_policy(**extra):
    return _make_serialisable(Compose([GaussianNoise(std=0.05, **extra)]).to_dict())


def test_phantoms_carry_demo_targets_and_state(session):
    state = session.state()
    assert state["loaded"] and state["ndim"] == 2 and state["axes"] == ["y", "x"]
    assert state["has_mask"] and state["summary"]["keypoints"] == 2
    assert "synthetic demo labels" in state["source"]
    plain = StudioSession()
    assert plain.state() == {"loaded": False, "policy": plain.policy}
    plain.load_phantom("dbt", demo_targets=False)
    assert plain.volume.mask is None and plain.state()["axes"] == ["z", "y", "x"]
    with pytest.raises(ValueError, match="Unknown phantom"):
        plain.load_phantom("pet")
    plain.close()


def test_preview_matches_training_seeding(session):
    policy = noise_policy()
    result = session.preview(
        {"policy": policy, "seed": 7, "sample_id": "case-1", "epoch": 2, "index": 1}
    )
    assert result["error"] is None and result["issues"] == []
    assert [e["action"] for e in result["trace"]] == ["applied", "applied"]
    expected = apply_for_sample(
        from_dict(policy), session.volume, seed=7, sample_id="case-1", epoch=2, augmentation_index=1
    )
    cached = session._cache[1]["output"]
    np.testing.assert_array_equal(cached.image, expected.image)
    assert result["input"]["png"].startswith("data:image/png")
    assert result["diff"] is not None
    assert "plain" not in result["input"] and "plain" not in result["output"]
    assert result["summary"]["output"]["shape"] == [256, 256]


def test_preview_cache_reuses_draw_until_inputs_change(session):
    request = {"policy": noise_policy(), "seed": 1}
    session.preview(request)
    first = session._cache
    session.preview({**request, "mask_alpha": 0.9, "window_mode": "auto"})
    assert session._cache is first
    session.preview({**request, "index": 1})
    assert session._cache is not first
    second = session._cache
    session.load_phantom("mri")
    assert session._cache is None
    session.preview(request)
    assert session._cache is not second


def test_preview_view_options_3d_and_validation(session):
    session.load_phantom("dbt")
    policy = _make_serialisable(Compose([RandomFlip(axes=("x",), p_per_axis=1.0)]).to_dict())
    result = session.preview({"policy": policy, "axis": "x", "position": 1.0, "mask_alpha": 0})
    assert result["input"]["index"] == 255 and result["input"]["length"] == 256
    assert result["output"]["targets"]["keypoints"] == []
    manual = session.preview({"policy": policy, "window_mode": "manual", "window": [0, 2]})
    assert manual["window"]["input"] == [0.0, 2.0]
    for bad, message in (
        ({"axis": "w"}, "axis"),
        ({"channel": "flair"}, "Unknown channel"),
        ({"mask_alpha": 2}, "mask_alpha"),
        ({"window_mode": "manual", "window": [1, 0]}, "low < high"),
        ({"window_mode": "manual", "window": "x"}, "numeric"),
        ({"window_mode": "lut"}, "window_mode"),
        ({"seed": -1}, "seed"),
        ({"epoch": 1.5}, "epoch"),
        ({"sample_id": [1]}, "sample_id"),
    ):
        with pytest.raises(ValueError, match=message):
            session.preview({"policy": policy, **bad})


def test_transform_errors_are_reported_not_raised(session):
    policy = _make_serialisable(Compose([RandomAffine(rotation=10)]).to_dict())
    session.load_phantom("dbt")
    from medaugmentx.transforms import CompressionVariation

    session.set_volume(MedVolume(session.volume.image[0]), "2D slice")
    bad = _make_serialisable(Compose([CompressionVariation()]).to_dict())
    result = session.preview({"policy": bad})
    assert result["output"] is None and result["error"]["type"] == "ValueError"
    assert session.preview({"policy": policy})["error"] is None


def test_policy_errors_point_at_the_failing_step(session):
    node = {
        "name": "Compose",
        "params": {
            "transforms": [
                {"name": "GaussianNoise", "params": {}},
                {
                    "name": "Guard",
                    "params": {
                        "transform": {"name": "RandomAffine", "params": {"rotation": [5, 1]}}
                    },
                },
            ]
        },
    }
    status = session.set_policy(node)
    assert status["ok"] is False and status["path"] == [1, "transform"]
    assert "rotation" in status["error"]
    assert session.policy == node
    with pytest.raises(PolicyError) as info:
        session.current_policy()
    assert info.value.path == [1, "transform"]
    with pytest.raises(PolicyError) as info:
        session.compile({"name": "Nope", "params": {}})
    assert info.value.path == [] and "not in the serialisation registry" in str(info.value)
    with pytest.raises(ValueError):
        session.set_policy([])
    ok = session.set_policy(noise_policy())
    assert ok["ok"] and "GaussianNoise" in ok["summary"]
    assert isinstance(session.current_policy(), Compose)


def test_new_steps_and_presets_are_valid(session):
    node = session.new_step("ClassBalancedCrop")
    assert node["params"]["size"] == [192, 192]
    assert session.new_step("OneOf")["params"]["transforms"][0]["name"] == "Compose"
    with pytest.raises(ValueError, match="Unknown transform"):
        session.new_step("Nope")
    preset = session.preset("ct", {"allow_flip": True, "unknown": True})
    assert from_dict(preset).to_dict()["params"]["transforms"][0]["params"]["p"] > 0
    with pytest.raises(ValueError, match="Unknown preset"):
        session.preset("pet")


def test_gallery_stress_and_record(session):
    policy = _make_serialisable(
        Compose([RandomFlip(axes=("x",), p_per_axis=1.0, p=0.5), GaussianNoise(std=0.01)]).to_dict()
    )
    gallery = session.gallery({"policy": policy, "count": 3, "index": 4})
    assert [item["index"] for item in gallery["items"]] == [4, 5, 6]
    assert all(item["view"]["png"] for item in gallery["items"])
    with pytest.raises(ValueError, match="count"):
        session.gallery({"policy": policy, "count": 0})

    report = session.stress({"policy": policy, "draws": 40})
    assert report["completed"] == 40 and report["passed"] == 40
    rates = {entry["path"]: entry["rate"] for entry in report["applied"]}
    assert rates["root.1"] == 1.0 and 0.2 < rates["root.0"] < 0.8
    strict = session.stress(
        {"policy": policy, "draws": 5, "validator": {"max_intensity_shift": 0.0}}
    )
    assert strict["rejected"] + strict["passed"] == 5
    with pytest.raises(ValueError, match="draws"):
        session.stress({"policy": policy, "draws": 0})

    record = session.record({"policy": policy, "seed": 3, "sample_id": 12})
    json.dumps(record)
    replayed = replay_augmentation(record, session.volume)
    expected = apply_for_sample(from_dict(policy), session.volume, seed=3, sample_id=12)
    np.testing.assert_array_equal(replayed.image, expected.image)


def test_export_and_import_round_trips(session):
    policy = Guard(
        PerChannel(Compose([GaussianNoise(std=0.1)]), shared=True),
        validator={"max_foreground_loss": 0.5},
        on_fail="retry",
        retries=3,
    )
    node = _make_serialisable(policy.to_dict())
    exported = session.export(node, "json")
    assert from_json(exported["text"]).to_dict() == policy.to_dict()
    yaml_text = session.export(node, "yaml")["text"]
    assert from_yaml(yaml_text).to_dict() == policy.to_dict()
    code = session.export(node, "python")["text"]
    namespace = {}
    exec(compile(code, "policy.py", "exec"), namespace)
    assert _make_serialisable(namespace["policy"].to_dict()) == node
    assert "max_foreground_loss" in code and "require_finite" not in code
    with pytest.raises(ValueError, match="format"):
        session.export(node, "toml")

    imported = session.import_text(exported["text"])
    assert imported["policy"] == node and imported["notices"] == []
    legacy = session.import_text(json.dumps(node))
    assert legacy["policy"] == node and "unversioned" in legacy["notices"][0]
    assert session.import_text(yaml_text)["policy"] == node
    with pytest.raises(ValueError):
        session.import_text("   ")


def test_python_snippet_handles_presets_and_custom_transforms(session):
    for preset in ("mri", "ct", "dxr", "dbt"):
        node = session.preset(preset)
        namespace = {}
        exec(python_snippet(node), namespace)
        assert namespace["policy"].to_dict() == from_dict(node).to_dict()

    from medaugmentx.core import Transform
    from medaugmentx.serialization import REGISTRY

    class StudioCustom(Transform):
        def apply(self, volume):
            return volume

    REGISTRY["StudioCustom"] = StudioCustom
    try:
        code = python_snippet({"name": "StudioCustom", "params": {"p": 1.0, "seed": None}})
        assert "register your custom transform StudioCustom" in code
        assert "from medaugmentx.transforms import" not in code
    finally:
        del REGISTRY["StudioCustom"]


def test_numpy_and_nifti_loading_with_masks_and_channels(tmp_path):
    nibabel = pytest.importorskip("nibabel")
    image = np.random.default_rng(0).random((6, 8, 10)).astype(np.float32)
    mask = (image > 0.5).astype(np.uint8)
    np.save(tmp_path / "image.npy", image)
    np.save(tmp_path / "mask.npy", mask)
    np.save(tmp_path / "other.npy", image * 2)
    session = StudioSession()
    try:
        state = session.load_path(str(tmp_path / "image.npy"), str(tmp_path / "mask.npy"))
        assert state["has_mask"] and state["summary"]["shape"] == [6, 8, 10]
        state = session.add_channel("t2", str(tmp_path / "other.npy"))
        assert state["channels"] == ["t2"]
        result = session.preview({"policy": noise_policy(), "channel": "t2"})
        assert result["output"] is not None
        with pytest.raises(ValueError, match="already exists"):
            session.add_channel("t2", str(tmp_path / "other.npy"))
        with pytest.raises(ValueError, match="Channel names"):
            session.add_channel("bad name!", str(tmp_path / "other.npy"))
        assert session.remove_channel("t2")["channels"] == []
        with pytest.raises(ValueError, match="No channel"):
            session.remove_channel("t2")

        affine = np.diag([0.5, 1.0, 2.0, 1.0])
        for name, array in (("img", image), ("seg", mask), ("flair", image * 3)):
            nifti = nibabel.Nifti1Image(np.ascontiguousarray(array.T), affine)
            nifti.header.set_xyzt_units("mm")
            nibabel.save(nifti, str(tmp_path / f"{name}.nii.gz"))
        state = session.load_path(str(tmp_path / "img.nii.gz"), str(tmp_path / "seg.nii.gz"))
        assert state["summary"]["geometry_known"] and state["source"] == "img.nii.gz"
        assert session.add_channel("flair", str(tmp_path / "flair.nii.gz"))["channels"] == ["flair"]
        with pytest.raises(ValueError, match="Channels must be"):
            session.add_channel("x", str(tmp_path / "x.txt"))
        with pytest.raises(FileNotFoundError):
            session.load_path(str(tmp_path / "missing.nii.gz"))
        with pytest.raises(FileNotFoundError):
            session.load_path(str(tmp_path / "img.nii.gz"), str(tmp_path / "nope.nii.gz"))
        with pytest.raises(ValueError, match="Masks are supported"):
            session.load_path(str(tmp_path), str(tmp_path / "seg.nii.gz"))

        stored = session.store_upload("image", "scan.nii.gz", b"data")
        assert stored.endswith(".nii.gz")
        with pytest.raises(ValueError, match="Uploads must"):
            session.upload_target("image", "scan.dcm")
        with pytest.raises(ValueError, match="role"):
            session.upload_target("other", "scan.npy")
    finally:
        session.close()


def test_session_accepts_in_memory_inputs():
    volume = MedVolume(np.zeros((4, 4), np.float32), channels={"b": np.ones((4, 4))})
    policy = Compose([GaussianNoise()])
    session = StudioSession(volume, policy, source="notebook array")
    assert session.state()["source"] == "notebook array"
    assert session.policy == _make_serialisable(policy.to_dict())
    with pytest.raises(TypeError):
        session.set_volume("not a volume", "x")
    with pytest.raises(PolicyError):
        StudioSession(policy={"name": "Nope", "params": {}})
    empty = StudioSession()
    with pytest.raises(ValueError, match="Load an image"):
        empty.preview({})
    with pytest.raises(ValueError, match="Load an image"):
        empty.add_channel("x", "y.npy")


def test_review_regressions_for_weights_seeds_rates_and_imports(session):
    assert session.new_step("OneOf")["params"]["weights"] is None
    preset = session.preset("mri")
    oneof = [t for t in preset["params"]["transforms"] if t["name"] == "OneOf"][0]
    assert oneof["params"]["weights"] is None
    oneof["params"]["transforms"].append(session.new_step("GaussianNoise"))
    assert session.set_policy(preset)["ok"]

    big = 2**64 + 13
    session.preview(
        {
            "policy": noise_policy(),
            "seed": str(big),
            "sample_id": "9007199254740993",
            "numeric_sample_id": True,
        }
    )
    expected = apply_for_sample(
        from_dict(noise_policy()), session.volume, seed=big, sample_id=9007199254740993
    )
    np.testing.assert_array_equal(session._cache[1]["output"].image, expected.image)
    for bad in (float("inf"), float("nan"), "1e3", "-1", " "):
        with pytest.raises(ValueError, match="seed"):
            session.preview({"policy": noise_policy(), "seed": bad})
    with pytest.raises(ValueError, match="count"):
        session.gallery({"policy": noise_policy(), "count": "3"})

    with_channel = StudioSession(
        MedVolume(np.random.default_rng(0).random((8, 8)), channels={"t2": np.ones((8, 8))})
    )
    per_channel = _make_serialisable(
        Compose([PerChannel(GaussianNoise(std=0.01)), Guard(GaussianNoise(std=0.01))]).to_dict()
    )
    rates = with_channel.stress({"policy": per_channel, "draws": 4})["applied"]
    assert rates and all(0.0 <= entry["rate"] <= 1.0 for entry in rates)
    with_channel.close()

    with pytest.raises(ValueError, match="JSON or YAML"):
        session.import_text("a: [")
    huge = _make_serialisable(Compose([GaussianNoise()], seed=2**63).to_dict())
    assert "2**53" in session.import_text(json.dumps(huge))["notices"][-1]


def test_masks_attach_to_any_loaded_image_and_uploads_are_pruned(tmp_path):
    nibabel = pytest.importorskip("nibabel")
    image = np.random.default_rng(2).random((6, 7, 8)).astype(np.float32)
    mask = (image > 0.5).astype(np.uint8)
    np.save(tmp_path / "mask.npy", mask)
    np.save(tmp_path / "wrong.npy", mask[:, :, :4])
    session = StudioSession(MedVolume(image, channels={"t2": image}), source="notebook")
    try:
        state = session.attach_mask(str(tmp_path / "mask.npy"))
        assert state["has_mask"] and state["channels"] == ["t2"]
        with pytest.raises(ValueError, match="does not match"):
            session.attach_mask(str(tmp_path / "wrong.npy"))
        with pytest.raises(ValueError, match="Masks must be"):
            session.attach_mask(__file__)

        affine = np.diag([1.0, 1.0, 2.0, 1.0])
        for name, array in (("img", image), ("seg", mask)):
            nifti = nibabel.Nifti1Image(np.ascontiguousarray(array.T), affine)
            nifti.header.set_xyzt_units("mm")
            nibabel.save(nifti, str(tmp_path / f"{name}.nii.gz"))
        far = affine.copy()
        far[0, 3] = 5.0
        shifted = nibabel.Nifti1Image(np.ascontiguousarray(mask.T), far)
        shifted.header.set_xyzt_units("mm")
        nibabel.save(shifted, str(tmp_path / "far.nii.gz"))
        session.load_path(str(tmp_path / "img.nii.gz"))
        session.add_channel("t2", str(tmp_path / "img.nii.gz"))
        state = session.attach_mask(str(tmp_path / "seg.nii.gz"))
        assert state["has_mask"] and state["channels"] == ["t2"]
        with pytest.raises(ValueError, match="same shape and patient-space grid"):
            session.attach_mask(str(tmp_path / "far.nii.gz"))

        first = session.store_upload("image", "a.npy", (tmp_path / "mask.npy").read_bytes())
        session.load_path(first)
        second = session.store_upload("image", "b.npy", (tmp_path / "mask.npy").read_bytes())
        session.load_path(second)
        assert sorted(os.listdir(session._uploads)) == [os.path.basename(second)]
    finally:
        session.close()
