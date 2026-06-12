"""Use a MedAugmentX pipeline with framework-style dataset samples."""

from __future__ import annotations

import numpy as np

from medaugmentx.interop import MonaiMapTransform, TorchIOTransform, TorchTransform
from medaugmentx.presets import mri_pipeline


class _Image:
    """Small TorchIO-shaped stand-in so this example has no TorchIO dependency."""

    def __init__(self, data: np.ndarray, spacing=(1.0, 0.7, 0.7)) -> None:
        self.data = data
        self.spacing = spacing

    def set_data(self, data: np.ndarray) -> None:
        self.data = data

    def copy(self):
        return self.__class__(self.data.copy(), spacing=self.spacing)


class _LabelMap(_Image):
    pass


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

    torchio_style_subject = {
        "t1": _Image(image),
        "seg": _LabelMap(mask),
        "case_id": "case-001",
    }

    torchio_augment = TorchIOTransform(
        mri_pipeline(seed=42),
        image_key="t1",
        label_key="seg",
        channel_dim=0,
    )
    out = torchio_augment(torchio_style_subject)
    print("TorchIO-style subject:", out["t1"].data.shape, out["seg"].data.dtype)


if __name__ == "__main__":
    main()
