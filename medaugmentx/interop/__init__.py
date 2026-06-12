"""Framework adapters for using MedAugmentX in training pipelines.

The adapters in this module are intentionally dependency-light: importing
``medaugmentx.interop`` does not import PyTorch, MONAI, or TorchIO. Tensor
and subject support is duck-typed and only touches optional frameworks when
their objects are passed at runtime.
"""

from medaugmentx.interop.adapters import (
    MonaiMapTransform,
    SampleTransform,
    TorchIOTransform,
    TorchTransform,
)

__all__ = ["SampleTransform", "TorchTransform", "MonaiMapTransform", "TorchIOTransform"]
