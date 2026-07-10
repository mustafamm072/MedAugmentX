"""Plausibility validation and safe-augmentation guards.

Augmentation is only useful if it produces *training-usable* volumes. A
transform that silently emits ``NaN``, collapses the dynamic range to a
constant, desynchronises the mask from the image, or crops away the only
labelled structure will not raise — it will just quietly poison the dataset,
and the damage usually surfaces weeks later as an underperforming model.

This module makes those failure modes loud. :class:`VolumeValidator` checks a
:class:`~medaugmentx.core.volume.MedVolume` against a configurable set of
clinical-plausibility rules, optionally comparing it to the pre-augmentation
reference volume. :class:`Guard` wraps any transform (or whole pipeline) and
validates its output on every call, so a single line makes an existing
pipeline fail-fast, warn, retry, or fall back to the untouched input.

Everything here depends only on NumPy, is fully seedable, and round-trips
through the same JSON/YAML serialisation as every other transform.

Scope: these rules are structural and statistical sanity checks. They catch
augmentation that is *obviously* unusable — they do not verify anatomical
correctness or clinical validity, and a passing report is not a substitute for
dataset review. Tune the thresholds to your normalisation and labelling
conventions.

Example::

    from medaugmentx import Compose, Guard, VolumeValidator
    from medaugmentx.transforms import RandomAffine, GammaCorrection

    pipeline = Guard(
        Compose([RandomAffine(), GammaCorrection()]),
        VolumeValidator(intensity_bounds=(0.0, 1.0), max_foreground_loss=0.5),
        on_fail="retry",
        retries=3,
        seed=42,
    )
    safe = pipeline(vol)   # NaN / collapsed / structure-destroying draws are retried
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from medaugmentx.core.base import Transform
from medaugmentx.core.utils import SeedLike, derive_rng
from medaugmentx.core.volume import MedVolume

Severity = str  # "error" | "warning"

ON_FAIL_MODES = ("raise", "warn", "revert", "retry")


class ValidationError(RuntimeError):
    """Raised when a :class:`Guard` in ``on_fail="raise"`` mode sees an error.

    The failing :class:`ValidationReport` is attached as :attr:`report`.
    """

    def __init__(self, report: ValidationReport) -> None:
        super().__init__(str(report))
        self.report = report


@dataclass(frozen=True)
class ValidationIssue:
    """A single problem found while validating a volume.

    Attributes:
        check: Name of the rule that produced the issue (e.g. ``"finite"``).
        severity: ``"error"`` (volume is unusable) or ``"warning"`` (volume is
            suspicious but not provably broken).
        message: Human-readable description, safe to log.
    """

    check: str
    severity: Severity
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check}: {self.message}"


@dataclass(frozen=True)
class ValidationReport:
    """The outcome of validating one volume.

    A report is *truthy* when the volume passed (no error-severity issues), so
    ``if report:`` reads naturally. Warnings never fail a report on their own.
    """

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def ok(self) -> bool:
        """``True`` when there are no error-severity issues."""
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        if not self.issues:
            return "ValidationReport: OK (0 issues)"
        header = (
            f"ValidationReport: {'OK' if self.ok else 'FAILED'} "
            f"({len(self.errors)} error(s), {len(self.warnings)} warning(s))"
        )
        return "\n".join([header, *(f"  {issue}" for issue in self.issues)])


class VolumeValidator:
    """Check a :class:`~medaugmentx.core.volume.MedVolume` for clinical plausibility.

    Each option toggles or parameterises one rule. Rules that compare against
    the pre-augmentation volume (``preserve_mask_labels``, ``max_foreground_loss``,
    ``max_intensity_shift``) are only evaluated when a ``reference`` is passed to
    :meth:`validate`; :class:`Guard` supplies it automatically.

    Args:
        require_finite: Fail if the image contains ``NaN`` or ``±Inf``.
        forbid_constant: Fail if the image collapsed to a single value (a common
            symptom of a broken intensity transform).
        check_mask_shape: Fail if a mask is present but no longer matches the
            image shape (guards against spatial ops that resize only one array).
        intensity_bounds: Optional ``(low, high)``. Warn if image values fall
            outside the expected range — useful for windowed CT or normalised
            inputs. Set ``strict_bounds=True`` to make it an error instead.
        strict_bounds: Escalate ``intensity_bounds`` violations to errors.
        min_foreground_fraction: Optional. Fail if the fraction of non-zero mask
            voxels drops below this absolute threshold.
        preserve_mask_labels: When a reference is given, fail if any label value
            present in the reference mask has completely disappeared (a structure
            was augmented out of existence).
        max_foreground_loss: When a reference is given, fail if the mask
            foreground shrank by more than this fraction (e.g. ``0.9`` allows up
            to a 90% reduction; a full wipe fails). ``None`` disables the check.
        max_intensity_shift: When a reference is given, warn if the image mean
            moved by more than this many reference standard deviations — a coarse
            distribution-drift signal.
    """

    def __init__(
        self,
        *,
        require_finite: bool = True,
        forbid_constant: bool = True,
        check_mask_shape: bool = True,
        intensity_bounds: tuple[float, float] | None = None,
        strict_bounds: bool = False,
        min_foreground_fraction: float | None = None,
        preserve_mask_labels: bool = True,
        max_foreground_loss: float | None = None,
        max_intensity_shift: float | None = None,
    ) -> None:
        if intensity_bounds is not None:
            lo, hi = intensity_bounds
            if not lo < hi:
                raise ValueError(f"intensity_bounds low must be < high, got {intensity_bounds}")
            intensity_bounds = (float(lo), float(hi))
        if min_foreground_fraction is not None and not 0.0 <= min_foreground_fraction <= 1.0:
            raise ValueError("min_foreground_fraction must be in [0, 1]")
        if max_foreground_loss is not None and not 0.0 <= max_foreground_loss <= 1.0:
            raise ValueError("max_foreground_loss must be in [0, 1]")
        if max_intensity_shift is not None and max_intensity_shift < 0:
            raise ValueError("max_intensity_shift must be non-negative")

        self.require_finite = bool(require_finite)
        self.forbid_constant = bool(forbid_constant)
        self.check_mask_shape = bool(check_mask_shape)
        self.intensity_bounds = intensity_bounds
        self.strict_bounds = bool(strict_bounds)
        self.min_foreground_fraction = min_foreground_fraction
        self.preserve_mask_labels = bool(preserve_mask_labels)
        self.max_foreground_loss = max_foreground_loss
        self.max_intensity_shift = max_intensity_shift

    def validate(
        self, volume: MedVolume, reference: MedVolume | None = None
    ) -> ValidationReport:
        """Run all enabled checks and return a :class:`ValidationReport`.

        Args:
            volume: The (augmented) volume to inspect.
            reference: Optional pre-augmentation volume. Comparative rules are
                skipped when it is ``None``.

        Returns:
            A report; ``report.ok`` is ``True`` when no errors were found.
        """
        if not isinstance(volume, MedVolume):
            raise TypeError(f"validate expects a MedVolume, got {type(volume).__name__}")

        issues: list[ValidationIssue] = []
        image = volume.image

        if self.require_finite and not np.all(np.isfinite(image)):
            n_bad = int(np.count_nonzero(~np.isfinite(image)))
            issues.append(
                ValidationIssue(
                    "finite", "error", f"image contains {n_bad} non-finite value(s) (NaN/Inf)"
                )
            )

        # A constant image is only diagnosable when the values are finite.
        if self.forbid_constant and np.all(np.isfinite(image)) and image.size > 0:
            if float(np.ptp(image)) == 0.0:
                issues.append(
                    ValidationIssue(
                        "constant",
                        "error",
                        f"image collapsed to a single constant value ({float(image.flat[0])})",
                    )
                )

        if self.check_mask_shape and volume.mask is not None:
            if volume.mask.shape != image.shape:
                issues.append(
                    ValidationIssue(
                        "mask_shape",
                        "error",
                        f"mask shape {volume.mask.shape} != image shape {image.shape}",
                    )
                )

        if self.intensity_bounds is not None and np.isfinite(image).any():
            lo, hi = self.intensity_bounds
            vmin, vmax = float(np.nanmin(image)), float(np.nanmax(image))
            if vmin < lo or vmax > hi:
                sev: Severity = "error" if self.strict_bounds else "warning"
                issues.append(
                    ValidationIssue(
                        "intensity_bounds",
                        sev,
                        f"image range [{vmin:.4g}, {vmax:.4g}] outside expected "
                        f"[{lo:.4g}, {hi:.4g}]",
                    )
                )

        if self.min_foreground_fraction is not None and volume.mask is not None:
            frac = float(np.count_nonzero(volume.mask)) / max(volume.mask.size, 1)
            if frac < self.min_foreground_fraction:
                issues.append(
                    ValidationIssue(
                        "min_foreground",
                        "error",
                        f"mask foreground fraction {frac:.4g} below minimum "
                        f"{self.min_foreground_fraction:.4g}",
                    )
                )

        if reference is not None:
            issues.extend(self._comparative_issues(volume, reference))

        return ValidationReport(tuple(issues))

    def _comparative_issues(
        self, volume: MedVolume, reference: MedVolume
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if (
            self.preserve_mask_labels
            and volume.mask is not None
            and reference.mask is not None
        ):
            before = set(np.unique(reference.mask).tolist())
            after = set(np.unique(volume.mask).tolist())
            lost = sorted(label for label in before - after if label != 0)
            if lost:
                issues.append(
                    ValidationIssue(
                        "mask_labels",
                        "error",
                        f"mask label(s) {lost} present before augmentation are now gone",
                    )
                )

        if (
            self.max_foreground_loss is not None
            and volume.mask is not None
            and reference.mask is not None
        ):
            before_fg = int(np.count_nonzero(reference.mask))
            after_fg = int(np.count_nonzero(volume.mask))
            if before_fg > 0:
                loss = 1.0 - (after_fg / before_fg)
                if loss > self.max_foreground_loss:
                    issues.append(
                        ValidationIssue(
                            "foreground_loss",
                            "error",
                            f"mask foreground shrank by {loss:.1%} "
                            f"(> {self.max_foreground_loss:.1%} allowed)",
                        )
                    )

        if self.max_intensity_shift is not None:
            ref_img = reference.image
            if np.isfinite(volume.image).all() and np.isfinite(ref_img).all():
                ref_std = float(np.std(ref_img))
                if ref_std > 0:
                    shift = abs(float(np.mean(volume.image)) - float(np.mean(ref_img))) / ref_std
                    if shift > self.max_intensity_shift:
                        issues.append(
                            ValidationIssue(
                                "intensity_shift",
                                "warning",
                                f"image mean shifted {shift:.2f}σ "
                                f"(> {self.max_intensity_shift:.2f}σ expected)",
                            )
                        )

        return issues

    def to_dict(self) -> dict[str, Any]:
        """Return the validator's configuration as a plain dict."""
        return {
            "require_finite": self.require_finite,
            "forbid_constant": self.forbid_constant,
            "check_mask_shape": self.check_mask_shape,
            "intensity_bounds": (
                list(self.intensity_bounds) if self.intensity_bounds is not None else None
            ),
            "strict_bounds": self.strict_bounds,
            "min_foreground_fraction": self.min_foreground_fraction,
            "preserve_mask_labels": self.preserve_mask_labels,
            "max_foreground_loss": self.max_foreground_loss,
            "max_intensity_shift": self.max_intensity_shift,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> VolumeValidator:
        params = dict(d)
        bounds = params.get("intensity_bounds")
        if bounds is not None:
            params["intensity_bounds"] = tuple(bounds)
        return cls(**params)

    def __repr__(self) -> str:
        opts = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"VolumeValidator({opts})"


class Guard(Transform):
    """Wrap a transform and validate its output on every call.

    ``Guard`` runs the wrapped transform, validates the result against the
    input volume, and then acts on any failure according to ``on_fail``:

    - ``"raise"``  — raise :class:`ValidationError` (fail loud; good for tests
      and one-off dataset audits).
    - ``"warn"``   — emit a :class:`UserWarning` and return the output anyway.
    - ``"revert"`` — discard the output and return the original volume
      unchanged (never inject a bad sample into training).
    - ``"retry"``  — re-run the wrapped transform with fresh randomness up to
      ``retries`` times; if none pass, fall back to reverting.

    ``Guard`` is itself a :class:`~medaugmentx.core.base.Transform`, so it
    nests inside ``Compose``/``OneOf``/``SomeOf`` and serialises like any other
    step.

    Args:
        transform: The transform or pipeline to guard.
        validator: A :class:`VolumeValidator`, its ``to_dict()`` mapping, or
            ``None`` for the default validator.
        on_fail: One of ``"raise"``, ``"warn"``, ``"revert"``, ``"retry"``.
        retries: Number of re-draws in ``"retry"`` mode (ignored otherwise).
        p: Probability of applying the wrapped transform at all.
        seed: Seed for the retry re-draw stream.
    """

    def __init__(
        self,
        transform: Transform,
        validator: VolumeValidator | Mapping[str, Any] | None = None,
        on_fail: str = "raise",
        retries: int = 1,
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if not isinstance(transform, Transform):
            raise TypeError(f"Guard expects a Transform, got {type(transform).__name__}")
        if on_fail not in ON_FAIL_MODES:
            raise ValueError(f"on_fail must be one of {ON_FAIL_MODES}, got {on_fail!r}")
        if int(retries) < 1:
            raise ValueError("retries must be >= 1")

        if validator is None:
            validator = VolumeValidator()
        elif isinstance(validator, Mapping):
            validator = VolumeValidator.from_dict(validator)
        elif not isinstance(validator, VolumeValidator):
            raise TypeError("validator must be a VolumeValidator, a mapping, or None")

        self.transform = transform
        self.validator = validator
        self.on_fail = on_fail
        self.retries = int(retries)
        # Reseed the child at construction so a standalone ``Guard(t, seed=...)``
        # is reproducible from its own seed, mirroring ``Compose``. This
        # deliberately overrides any seed the wrapped transform was built with.
        self._reseed_child()

    def _reseed_child(self) -> None:
        (child_rng,) = derive_rng(self.rng, 1)
        self.transform.set_rng(child_rng)

    def set_rng(self, rng: np.random.Generator) -> None:
        # Give the wrapped transform its own derived stream, mirroring Compose.
        super().set_rng(rng)
        self._reseed_child()

    def apply(self, volume: MedVolume) -> MedVolume:
        reference = volume
        attempts = self.retries if self.on_fail == "retry" else 1
        last_report: ValidationReport | None = None

        for attempt in range(attempts):
            if attempt > 0:
                # Fresh randomness for each retry, deterministic given the seed.
                self._reseed_child()
            candidate = self.transform(reference)
            report = self.validator.validate(candidate, reference)
            if report.ok:
                return candidate
            last_report = report

        assert last_report is not None
        return self._handle_failure(reference, candidate, last_report)

    def _handle_failure(
        self, reference: MedVolume, candidate: MedVolume, report: ValidationReport
    ) -> MedVolume:
        if self.on_fail == "raise":
            raise ValidationError(report)
        if self.on_fail == "warn":
            warnings.warn(f"Guard validation failed:\n{report}", UserWarning, stacklevel=2)
            return candidate
        # "revert" and exhausted "retry" both fall back to the untouched input.
        if self.on_fail == "retry":
            warnings.warn(
                f"Guard exhausted {self.retries} retries; reverting to input.\n{report}",
                UserWarning,
                stacklevel=2,
            )
        return reference

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "Guard",
            "params": {
                "transform": self.transform.to_dict(),
                "validator": self.validator.to_dict(),
                "on_fail": self.on_fail,
                "retries": self.retries,
                "p": self.p,
                "seed": self._seed,
            },
        }


__all__ = [
    "ValidationError",
    "ValidationIssue",
    "ValidationReport",
    "VolumeValidator",
    "Guard",
]
