"""Use a MedAugmentX pipeline with framework-style dataset samples."""

from __future__ import annotations

import numpy as np

from medaugmentx.interop import MonaiMapTransform, TorchTransform
from medaugmentx.presets import mri_pipeline


def main() -> None:
    image = np.random.default_rng(0).random((1, 16, 32, 32), dtype=np.float32)
    mask = (image > 0.7).astype(np.uint8)

    torch_style_sample = {
        "image": image,
        "mask": mask,
        "spacing": (1.0, 0.7, 0.7),
        "metadata": {"modality": "MR"},
    }

    augment = TorchTransform(mri_pipeline(seed=42), channel_dim=0)
    out = augment(torch_style_sample)
    print("Torch-style sample:", out["image"].shape, out["mask"].dtype)

    monai_style_sample = {
        "image": image,
        "label": mask,
        "spacing": (1.0, 0.7, 0.7),
        "metadata": {"modality": "MR"},
    }

    monai_augment = MonaiMapTransform(mri_pipeline(seed=42), channel_dim=0)
    out = monai_augment(monai_style_sample)
    print("MONAI-style sample:", out["image"].shape, out["label"].dtype)


if __name__ == "__main__":
    main()
