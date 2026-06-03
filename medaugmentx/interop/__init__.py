"""Framework adapters for using MedAugmentX in training pipelines.

The adapters in this module are intentionally dependency-light: importing
``medaugmentx.interop`` does not import PyTorch or MONAI. Tensor
support is duck-typed and only touches PyTorch when a torch tensor is passed
at runtime.
"""

from medaugmentx.interop.adapters import MonaiMapTransform, SampleTransform, TorchTransform

__all__ = ["SampleTransform", "TorchTransform", "MonaiMapTransform"]
