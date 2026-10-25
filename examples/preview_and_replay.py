"""Inspect image/mask draws and reproduce one augmentation without downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from medaugmentx import (
    Guard,
    VolumeValidator,
    apply_for_sample,
    preview_augmentation,
    record_augmentation,
    replay_augmentation,
)
from medaugmentx.phantoms import mri_phantom
from medaugmentx.presets import mri_pipeline
from medaugmentx.serialization import to_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("preview-output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    volume = mri_phantom(size=64)
    # Synthetic foreground label for demonstrating image/mask registration.
    volume = volume.replace(mask=(volume.image > 0.2).astype(np.uint8))
    policy = Guard(
        mri_pipeline(seed=42, normalize=True),
        VolumeValidator(max_foreground_loss=0.25),
        on_fail="retry",
        retries=3,
        seed=42,
    )
    report = preview_augmentation(policy, volume, str(args.output / "preview.png"), samples=6)
    (args.output / "qa.json").write_text(json.dumps(report, indent=2))
    (args.output / "policy.json").write_text(to_json(policy))
    out, record = record_augmentation(policy, volume)
    restored = replay_augmentation(json.loads(json.dumps(record)), volume)
    np.testing.assert_array_equal(out.image, restored.image)
    (args.output / "replay.json").write_text(json.dumps(record, indent=2))
    # In Dataset.__getitem__, use a stable dataset ID and update epoch explicitly.
    deterministic = apply_for_sample(policy, volume, seed=42, sample_id="synthetic-001", epoch=3)
    assert deterministic.shape == volume.shape
    print(f"Saved preview, QA report, policy and replay record to {args.output}")


if __name__ == "__main__":
    main()
