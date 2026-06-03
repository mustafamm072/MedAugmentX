"""Lightweight adapters for PyTorch, torchvision, and MONAI-style samples."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Union

import numpy as np

from medaugmentx.core import MedVolume, Transform

ChannelDim = Union[int, Literal["auto", "first", "last"], None]


@dataclass(frozen=True)
class _ArraySpec:
    """How to restore an augmented NumPy array to the caller's input type."""

    template: Any
    kind: Literal["numpy", "torch"]
    channel_axis: int | None
    preserve_dtype: bool


def _is_torch_tensor(value: Any) -> bool:
    """Return True for torch tensors without importing torch at module import."""
    cls = type(value)
    return (
        cls.__module__.startswith("torch")
        and hasattr(value, "detach")
        and hasattr(value, "cpu")
        and hasattr(value, "numpy")
    )


def _as_numpy(value: Any) -> tuple[np.ndarray, Literal["numpy", "torch"]]:
    if isinstance(value, np.ndarray):
        return value, "numpy"
    if _is_torch_tensor(value):
        return value.detach().cpu().numpy(), "torch"
    raise TypeError(
        "Expected a numpy.ndarray, torch.Tensor, MedVolume, tuple, or mapping sample; "
        f"got {type(value).__name__}"
    )


def _normalise_channel_axis(channel_dim: ChannelDim, ndim: int) -> int | Literal["auto"] | None:
    if channel_dim is None:
        return None
    if channel_dim == "auto":
        return "auto"
    if channel_dim == "first":
        return 0
    if channel_dim == "last":
        return ndim - 1
    axis = int(channel_dim)
    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        raise ValueError(f"channel_dim={channel_dim!r} is out of range for ndim={ndim}")
    return axis


def _strip_channel(array: np.ndarray, channel_dim: ChannelDim) -> tuple[np.ndarray, int | None]:
    """Remove one singleton channel axis so MedVolume sees a 2D/3D image."""
    axis = _normalise_channel_axis(channel_dim, array.ndim)
    if axis is None:
        if array.ndim in (2, 3):
            return array, None
        raise ValueError(
            "MedVolume supports 2D or 3D arrays. Pass channel_dim=0, "
            "channel_dim='last', or channel_dim='auto' for singleton-channel tensors."
        )

    if axis == "auto":
        if array.ndim in (3, 4) and array.shape[0] == 1:
            return np.squeeze(array, axis=0), 0
        if array.ndim in (3, 4) and array.shape[-1] == 1:
            return np.squeeze(array, axis=array.ndim - 1), array.ndim - 1
        if array.ndim in (2, 3):
            return array, None
        raise ValueError(
            "Could not infer a singleton channel dimension. Pass channel_dim explicitly."
        )

    if array.shape[axis] != 1:
        raise ValueError(
            "MedAugmentX transforms operate on single-channel 2D/3D medical images; "
            f"channel axis {axis} has size {array.shape[axis]}."
        )
    return np.squeeze(array, axis=axis), axis


def _to_volume_array(
    value: Any,
    *,
    channel_dim: ChannelDim,
    preserve_dtype: bool,
) -> tuple[np.ndarray, _ArraySpec]:
    array, kind = _as_numpy(value)
    squeezed, channel_axis = _strip_channel(array, channel_dim)
    return squeezed, _ArraySpec(value, kind, channel_axis, preserve_dtype)


def _restore_channel(array: np.ndarray, channel_axis: int | None) -> np.ndarray:
    if channel_axis is None:
        return array
    return np.expand_dims(array, axis=channel_axis)


def _restore_array(array: np.ndarray, spec: _ArraySpec) -> Any:
    restored = _restore_channel(array, spec.channel_axis)
    if spec.kind == "numpy":
        if spec.preserve_dtype:
            return restored.astype(spec.template.dtype, copy=False)
        return restored

    import torch

    out = torch.as_tensor(restored, device=spec.template.device)
    if spec.preserve_dtype:
        out = out.to(dtype=spec.template.dtype)
    return out


class SampleTransform:
    """Adapt a MedAugmentX transform to common dataset sample shapes.

    ``SampleTransform`` accepts one of:

    - a :class:`~medaugmentx.core.MedVolume`, returned as a ``MedVolume``;
    - an image array/tensor, returned as the same array/tensor kind;
    - ``(image, mask)``, returned as the same tuple/list kind;
    - a mapping with image/mask keys, returned as a shallow-copied mapping.

    PyTorch is optional. If a torch tensor is passed, the adapter moves it to
    CPU for the NumPy/SciPy transform, then restores the result to the original
    device. Masks preserve their dtype by default; image dtype follows the
    transform output unless ``preserve_image_dtype=True`` is set.
    """

    def __init__(
        self,
        transform: Transform,
        *,
        image_key: str = "image",
        mask_key: str | None = "mask",
        spacing_key: str | None = "spacing",
        metadata_key: str | None = "metadata",
        channel_dim: ChannelDim = "auto",
        preserve_image_dtype: bool = False,
        preserve_mask_dtype: bool = True,
    ) -> None:
        if not isinstance(transform, Transform):
            raise TypeError(
                f"SampleTransform expects a medaugmentx Transform; got {type(transform).__name__}"
            )
        self.transform = transform
        self.image_key = image_key
        self.mask_key = mask_key
        self.spacing_key = spacing_key
        self.metadata_key = metadata_key
        self.channel_dim = channel_dim
        self.preserve_image_dtype = bool(preserve_image_dtype)
        self.preserve_mask_dtype = bool(preserve_mask_dtype)

    def __call__(self, sample: Any) -> Any:
        if isinstance(sample, MedVolume):
            return self.transform(sample)
        if isinstance(sample, Mapping):
            return self._call_mapping(sample)
        if isinstance(sample, tuple) or isinstance(sample, list):
            return self._call_sequence(sample)
        image, _ = self._augment(image=sample, mask=None)
        return image

    def _augment(
        self,
        *,
        image: Any,
        mask: Any | None,
        spacing: tuple[float, ...] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any | None]:
        image_array, image_spec = _to_volume_array(
            image,
            channel_dim=self.channel_dim,
            preserve_dtype=self.preserve_image_dtype,
        )
        mask_array = None
        mask_spec = None
        if mask is not None:
            mask_array, mask_spec = _to_volume_array(
                mask,
                channel_dim=self.channel_dim,
                preserve_dtype=self.preserve_mask_dtype,
            )

        volume = MedVolume(
            image=image_array,
            mask=mask_array,
            spacing=() if spacing is None else tuple(float(s) for s in spacing),
            metadata={} if metadata is None else dict(metadata),
        )
        augmented = self.transform(volume)
        out_image = _restore_array(augmented.image, image_spec)
        out_mask = None
        if mask_spec is not None and augmented.mask is not None:
            out_mask = _restore_array(augmented.mask, mask_spec)
        return out_image, out_mask

    def _call_mapping(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.image_key not in sample:
            raise KeyError(f"image_key={self.image_key!r} not found in sample")
        mask_key = self.mask_key
        mask_present = mask_key is not None and mask_key in sample
        spacing = self._get_optional_tuple(sample, self.spacing_key)
        metadata = self._get_optional_mapping(sample, self.metadata_key)
        image, mask = self._augment(
            image=sample[self.image_key],
            mask=sample[mask_key] if mask_present and mask_key is not None else None,
            spacing=spacing,
            metadata=metadata,
        )
        out = dict(sample)
        out[self.image_key] = image
        if mask_present and mask_key is not None:
            out[mask_key] = mask
        return out

    def _call_sequence(self, sample: tuple[Any, ...] | list[Any]) -> tuple[Any, Any] | list[Any]:
        if len(sample) != 2:
            raise ValueError("Sequence samples must be (image, mask)")
        image, mask = self._augment(image=sample[0], mask=sample[1])
        if isinstance(sample, list):
            return [image, mask]
        return image, mask

    @staticmethod
    def _get_optional_tuple(
        sample: Mapping[str, Any],
        key: str | None,
    ) -> tuple[float, ...] | None:
        if key is None or key not in sample or sample[key] is None:
            return None
        return tuple(float(s) for s in sample[key])

    @staticmethod
    def _get_optional_mapping(
        sample: Mapping[str, Any],
        key: str | None,
    ) -> Mapping[str, Any] | None:
        if key is None or key not in sample or sample[key] is None:
            return None
        value = sample[key]
        if not isinstance(value, Mapping):
            raise TypeError(f"metadata_key={key!r} must point to a mapping")
        return value

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(transform={self.transform!r}, "
            f"image_key={self.image_key!r}, mask_key={self.mask_key!r}, "
            f"channel_dim={self.channel_dim!r})"
        )


class TorchTransform(SampleTransform):
    """PyTorch/torchvision-compatible callable wrapper for MedAugmentX transforms."""


class MonaiMapTransform(SampleTransform):
    """MONAI-style dictionary wrapper using ``image`` and ``label`` keys by default."""

    def __init__(
        self,
        transform: Transform,
        *,
        image_key: str = "image",
        label_key: str | None = "label",
        spacing_key: str | None = "spacing",
        metadata_key: str | None = "metadata",
        channel_dim: ChannelDim = "auto",
        preserve_image_dtype: bool = False,
        preserve_label_dtype: bool = True,
    ) -> None:
        super().__init__(
            transform,
            image_key=image_key,
            mask_key=label_key,
            spacing_key=spacing_key,
            metadata_key=metadata_key,
            channel_dim=channel_dim,
            preserve_image_dtype=preserve_image_dtype,
            preserve_mask_dtype=preserve_label_dtype,
        )
