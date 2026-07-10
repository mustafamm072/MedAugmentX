"""Safe augmentation with the plausibility validator (MedAugmentX 0.8.0).

Augmentation can silently produce training-unusable volumes: NaN pixels,
intensity that collapses to a constant, a mask that no longer lines up with the
image, or a spatial draw that crops the only labelled structure out of frame.
These never raise on their own — they just quietly degrade the dataset.

``VolumeValidator`` audits a volume against clinical-plausibility rules, and
``Guard`` wraps any transform or pipeline so bad draws are caught on every call
and either raised, warned, reverted, or retried.

Run with:

    python examples/safe_augmentation.py
"""
from __future__ import annotations

import numpy as np

from medaugmentx import Compose, Guard, MedVolume, Transform, VolumeValidator
from medaugmentx.serialization import from_json, to_json
from medaugmentx.transforms import GammaCorrection, RandomAffine


class BrokenNormalise(Transform):
    """A deliberately buggy transform: divides by zero range -> NaN everywhere."""

    def apply(self, volume: MedVolume) -> MedVolume:
        img = volume.image
        with np.errstate(divide="ignore", invalid="ignore"):
            bad = (img - img.min()) / (img.max() - img.max())
        return volume.replace(image=bad)


def main() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((24, 128, 128), dtype=np.float64).astype(np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[6:18, 32:96, 32:96] = 1
    vol = MedVolume(image=image, mask=mask, spacing=(1.0, 0.5, 0.5), metadata={"modality": "MR"})

    # 1. Validate a volume directly and read the report.
    validator = VolumeValidator(intensity_bounds=(0.0, 1.0), max_foreground_loss=0.6)
    print("Clean volume:")
    print(validator.validate(vol))

    broken = BrokenNormalise().apply(vol)
    print("\nBroken (NaN) volume:")
    print(validator.validate(broken, reference=vol))

    # 2. Guard a healthy pipeline — output passes straight through.
    safe = Guard(
        Compose([RandomAffine(), GammaCorrection()]),
        validator,
        on_fail="retry",
        retries=5,
        seed=42,
    )
    out = safe(vol)
    print(f"\nGuarded pipeline output: shape={out.shape}, finite={np.isfinite(out.image).all()}")

    # 3. Guard a broken transform in 'revert' mode — the input is returned untouched
    #    instead of injecting a NaN sample into training.
    guarded_broken = Guard(BrokenNormalise(), validator, on_fail="revert")
    reverted = guarded_broken(vol)
    print(f"Reverted on failure: identical to input = {np.array_equal(reverted.image, vol.image)}")

    # 4. Guards serialise like any other transform.
    restored = from_json(to_json(safe))
    print(f"\nRound-tripped guard: {type(restored).__name__}, on_fail={restored.on_fail!r}")


if __name__ == "__main__":
    main()
