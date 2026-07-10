"""Tests for the plausibility validator and the Guard transform."""
from __future__ import annotations

import numpy as np
import pytest

from medaugmentx import (
    Compose,
    Guard,
    MedVolume,
    ValidationError,
    ValidationReport,
    VolumeValidator,
)
from medaugmentx.core import Transform
from medaugmentx.serialization import from_json, to_json

# --------------------------------------------------------------------------- #
# Test doubles                                                                 #
# --------------------------------------------------------------------------- #


class MakeNaN(Transform):
    """Deterministically poison the image with NaN — always invalid."""

    def apply(self, volume):
        img = volume.image.copy()
        img[0, 0] = np.nan
        return volume.replace(image=img)


class WipeMask(Transform):
    """Zero out the whole mask — destroys every labelled structure."""

    def apply(self, volume):
        return volume.replace(mask=np.zeros_like(volume.mask))


class FlakyNaN(Transform):
    """Emit NaN unless the RNG rolls above a threshold, so retries can recover."""

    def apply(self, volume):
        if self.rng.random() < 0.5:
            img = volume.image.copy()
            img[0, 0] = np.nan
            return volume.replace(image=img)
        return volume


# --------------------------------------------------------------------------- #
# VolumeValidator                                                              #
# --------------------------------------------------------------------------- #


def test_clean_volume_passes(vol3d):
    report = VolumeValidator().validate(vol3d)
    assert report.ok
    assert bool(report) is True
    assert report.issues == ()


def test_nan_is_an_error(vol2d):
    bad = MakeNaN().apply(vol2d)
    report = VolumeValidator().validate(bad)
    assert not report.ok
    assert any(i.check == "finite" and i.severity == "error" for i in report.issues)


def test_constant_image_is_an_error():
    vol = MedVolume(image=np.full((8, 8), 0.3, dtype=np.float32))
    report = VolumeValidator().validate(vol)
    assert not report.ok
    assert any(i.check == "constant" for i in report.errors)


def test_intensity_bounds_warns_then_errors():
    vol = MedVolume(image=np.linspace(-2, 5, 64, dtype=np.float32).reshape(8, 8))

    warn_report = VolumeValidator(intensity_bounds=(0.0, 1.0)).validate(vol)
    assert warn_report.ok  # warning only
    assert any(i.check == "intensity_bounds" and i.severity == "warning" for i in warn_report.issues)

    err_report = VolumeValidator(
        intensity_bounds=(0.0, 1.0), strict_bounds=True
    ).validate(vol)
    assert not err_report.ok


def test_min_foreground_fraction(vol3d):
    # vol3d mask is sparse; demand an implausibly large foreground.
    report = VolumeValidator(min_foreground_fraction=0.9).validate(vol3d)
    assert any(i.check == "min_foreground" for i in report.errors)


def test_preserve_mask_labels_detects_wiped_structure(vol3d):
    wiped = WipeMask().apply(vol3d)
    report = VolumeValidator().validate(wiped, reference=vol3d)
    assert any(i.check == "mask_labels" for i in report.errors)


def test_max_foreground_loss(vol3d):
    wiped = WipeMask().apply(vol3d)
    report = VolumeValidator(
        preserve_mask_labels=False, max_foreground_loss=0.5
    ).validate(wiped, reference=vol3d)
    assert any(i.check == "foreground_loss" for i in report.errors)


def test_comparative_checks_skipped_without_reference(vol3d):
    wiped = WipeMask().apply(vol3d)
    # No reference -> label preservation cannot fire.
    report = VolumeValidator().validate(wiped)
    assert not any(i.check == "mask_labels" for i in report.issues)


def test_intensity_shift_warns():
    ref = MedVolume(image=np.random.default_rng(0).random((16, 16)).astype(np.float32))
    shifted = ref.replace(image=ref.image + 10.0)
    report = VolumeValidator(max_intensity_shift=1.0).validate(shifted, reference=ref)
    assert any(i.check == "intensity_shift" and i.severity == "warning" for i in report.issues)


def test_validator_rejects_bad_config():
    with pytest.raises(ValueError):
        VolumeValidator(intensity_bounds=(1.0, 0.0))
    with pytest.raises(ValueError):
        VolumeValidator(max_foreground_loss=1.5)


def test_report_str_is_readable(vol2d):
    report = VolumeValidator().validate(MakeNaN().apply(vol2d))
    text = str(report)
    assert "FAILED" in text
    assert "finite" in text


# --------------------------------------------------------------------------- #
# Guard                                                                        #
# --------------------------------------------------------------------------- #


def test_guard_raise_mode(vol2d):
    guard = Guard(MakeNaN(), on_fail="raise")
    with pytest.raises(ValidationError) as exc:
        guard(vol2d)
    assert isinstance(exc.value.report, ValidationReport)


def test_guard_revert_mode_returns_input(vol2d):
    guard = Guard(MakeNaN(), on_fail="revert")
    out = guard(vol2d)
    assert np.array_equal(out.image, vol2d.image)


def test_guard_warn_mode_returns_output(vol2d):
    guard = Guard(MakeNaN(), on_fail="warn")
    with pytest.warns(UserWarning):
        out = guard(vol2d)
    assert not np.all(np.isfinite(out.image))  # bad output passed through


def test_guard_retry_recovers(vol2d):
    guard = Guard(FlakyNaN(), on_fail="retry", retries=25, seed=1)
    out = guard(vol2d)
    assert np.all(np.isfinite(out.image))


def test_guard_retry_exhaustion_reverts(vol2d):
    guard = Guard(MakeNaN(), on_fail="retry", retries=3, seed=1)
    with pytest.warns(UserWarning):
        out = guard(vol2d)
    assert np.array_equal(out.image, vol2d.image)


def test_guard_passes_valid_output_through(vol3d):
    from medaugmentx.transforms import GammaCorrection

    guard = Guard(GammaCorrection(gamma=(0.9, 1.1)), on_fail="raise", seed=7)
    out = guard(vol3d)
    assert out.shape == vol3d.shape


def test_guard_is_deterministic(vol2d):
    g1 = Guard(FlakyNaN(), on_fail="retry", retries=25, seed=42)
    g2 = Guard(FlakyNaN(), on_fail="retry", retries=25, seed=42)
    assert np.array_equal(g1(vol2d).image, g2(vol2d).image)


def test_standalone_guard_reseeds_child_from_its_own_seed(vol2d):
    # A standalone Guard must be reproducible from its own seed even when the
    # wrapped transform actually modifies the image (mirrors Compose seeding).
    from medaugmentx.transforms import GaussianNoise

    a = Guard(GaussianNoise(std=0.05), on_fail="warn", seed=42)(vol2d)
    b = Guard(GaussianNoise(std=0.05), on_fail="warn", seed=42)(vol2d)
    assert np.array_equal(a.image, b.image)

    c = Guard(GaussianNoise(std=0.05), on_fail="warn", seed=7)(vol2d)
    assert not np.array_equal(a.image, c.image)


def test_guard_rejects_bad_args(vol2d):
    with pytest.raises(TypeError):
        Guard("not a transform")
    with pytest.raises(ValueError):
        Guard(MakeNaN(), on_fail="explode")
    with pytest.raises(ValueError):
        Guard(MakeNaN(), on_fail="retry", retries=0)


def test_guard_nests_in_compose(vol3d):
    from medaugmentx.transforms import GammaCorrection

    pipeline = Compose(
        [Guard(GammaCorrection(), on_fail="revert")],
        seed=3,
    )
    out = pipeline(vol3d)
    assert out.shape == vol3d.shape


# --------------------------------------------------------------------------- #
# Serialisation round-trip                                                     #
# --------------------------------------------------------------------------- #


def test_guard_json_round_trip():
    from medaugmentx.transforms import GaussianNoise

    guard = Guard(
        GaussianNoise(std=0.05),
        VolumeValidator(intensity_bounds=(0.0, 1.0), max_foreground_loss=0.5),
        on_fail="retry",
        retries=4,
        seed=11,
    )
    restored = from_json(to_json(guard))
    assert isinstance(restored, Guard)
    assert restored.on_fail == "retry"
    assert restored.retries == 4
    assert restored.validator.to_dict() == guard.validator.to_dict()
    assert restored.transform.to_dict() == guard.transform.to_dict()


def test_guard_in_compose_round_trip():
    from medaugmentx.transforms import GammaCorrection

    pipeline = Compose([Guard(GammaCorrection(), on_fail="warn")], seed=5)
    restored = from_json(to_json(pipeline))
    assert isinstance(restored, Compose)
    assert isinstance(restored.transforms[0], Guard)
