"""Import-lazy adapters with explicit spatial layout and patient-space geometry."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Union

import numpy as np

from medaugmentx.core import MedVolume, Transform
from medaugmentx.core.physical import axis_permutation

ChannelDim = Union[int, Literal["auto", "first", "last"], None]


@dataclass(frozen=True)
class _ArraySpec:
    template: Any
    kind: str
    channel_axis: int | None
    preserve_dtype: bool
    reverse: bool = False


def _is_torch_tensor(value: Any) -> bool:
    return any(
        cls.__module__.startswith("torch") and cls.__name__ in ("Tensor", "TensorBase")
        for cls in type(value).__mro__
    )


def _as_numpy(value: Any) -> tuple[np.ndarray, str]:
    if isinstance(value, np.ndarray):
        return value, "numpy"
    if _is_torch_tensor(value):
        tensor = value.detach().cpu()
        if str(tensor.dtype) == "torch.bfloat16":
            tensor = tensor.float()
        return tensor.numpy(), "torch"
    raise TypeError(
        f"Expected a numpy.ndarray, torch.Tensor or supported sample; got {type(value).__name__}"
    )


def _normalise_channel_axis(channel_dim: ChannelDim, ndim: int) -> int | str | None:
    if channel_dim in (None, "auto"):
        return channel_dim
    if channel_dim == "first":
        return 0
    if channel_dim == "last":
        return ndim - 1
    if not isinstance(channel_dim, int):
        raise ValueError("channel_dim must be None, auto, first, last, or an integer")
    axis = channel_dim + ndim if channel_dim < 0 else channel_dim
    if not 0 <= axis < ndim:
        raise ValueError("channel_dim is out of range")
    return axis


def _to_volume_array(
    value: Any, *, channel_dim: ChannelDim, preserve_dtype: bool, reverse: bool = False
) -> tuple[np.ndarray, _ArraySpec]:
    array, kind = _as_numpy(value)
    axis = _normalise_channel_axis(channel_dim, array.ndim)
    if axis == "auto":
        axis = (
            0
            if array.ndim in (3, 4) and array.shape[0] == 1
            else array.ndim - 1
            if array.ndim in (3, 4) and array.shape[-1] == 1
            else None
        )
    assert axis is None or isinstance(axis, int)
    if axis is not None:
        if array.shape[axis] != 1:
            raise ValueError(
                "MedAugmentX supports single-channel images; channel axis must be singleton"
            )
        array = np.squeeze(array, axis=axis)
    if array.ndim not in (2, 3):
        raise ValueError("Expected 2D/3D spatial data; configure a singleton channel_dim")
    if reverse:
        array = array.transpose()
    return array, _ArraySpec(value, kind, axis, preserve_dtype, reverse)


def _restore_array(array: np.ndarray, spec: _ArraySpec, affine: np.ndarray | None = None) -> Any:
    out = array.transpose() if spec.reverse else array
    if spec.channel_axis is not None:
        out = np.expand_dims(out, spec.channel_axis)
    if spec.kind == "numpy":
        if spec.preserve_dtype:
            if spec.template.dtype.kind in "iu":
                bounds = np.iinfo(spec.template.dtype)
                out = np.clip(out, bounds.min, bounds.max)
            out = out.astype(spec.template.dtype, copy=False)
        return np.ascontiguousarray(out)
    import torch

    tensor = torch.as_tensor(np.ascontiguousarray(out), device=spec.template.device)
    if spec.preserve_dtype:
        if not spec.template.dtype.is_floating_point and spec.template.dtype != torch.bool:
            bounds = torch.iinfo(spec.template.dtype)
            tensor = tensor.clamp(bounds.min, bounds.max)
        tensor = tensor.to(spec.template.dtype)
    if hasattr(spec.template, "applied_operations") and hasattr(spec.template, "affine"):
        from monai.data import MetaTensor

        # Our transforms are not MONAI-invertible. Do not retain stale inversion
        # history that claims an upstream grid still matches this tensor.
        metadata = copy.deepcopy(spec.template.meta)
        metadata.pop("affine", None)
        metadata["medaugmentx_noninvertible"] = True
        tensor = MetaTensor(tensor, affine=affine, meta=metadata, applied_operations=[])
    return tensor


class SampleTransform:
    """Adapt one image plus labels, targets, spacing, and affine in dataset samples.

    Spatial order is explicit: ``array`` means DHW/HW, ``native`` means XYZ/XY.
    Affines use the sample's spatial order and RAS millimetres. Arrays without
    affine have unknown patient geometry. Tensor processing runs on CPU and
    restores the input device; this is not a GPU backend.
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
        spatial_order: str = "array",
        affine_key: str | None = "affine",
        require_mask: bool = False,
    ) -> None:
        if not isinstance(transform, Transform):
            raise TypeError("SampleTransform expects a medaugmentx Transform")
        if spatial_order not in ("array", "native"):
            raise ValueError("spatial_order must be array or native")
        self.transform = transform
        self.image_key, self.mask_key = image_key, mask_key
        self.spacing_key, self.metadata_key = spacing_key, metadata_key
        self.channel_dim = channel_dim
        self.preserve_image_dtype, self.preserve_mask_dtype = (
            preserve_image_dtype,
            preserve_mask_dtype,
        )
        self.spatial_order, self.affine_key, self.require_mask = (
            spatial_order,
            affine_key,
            require_mask,
        )

    def __call__(self, sample: Any) -> Any:
        if isinstance(sample, MedVolume):
            return self.transform(sample)
        if isinstance(sample, Mapping):
            return self._call_mapping(sample)
        if isinstance(sample, (tuple, list)):
            if len(sample) != 2:
                raise ValueError("Sequence samples must be (image, mask)")
            image, mask, _ = self._augment(image=sample[0], mask=sample[1])
            return [image, mask] if isinstance(sample, list) else (image, mask)
        return self._augment(image=sample, mask=None)[0]

    def _augment(
        self,
        *,
        image: Any,
        mask: Any | None,
        spacing: Any = None,
        metadata: Mapping[str, Any] | None = None,
        affine: Any = None,
        targets: Mapping[str, Any] | None = None,
        reverse: bool | None = None,
    ) -> tuple[Any, Any | None, MedVolume]:
        if reverse is None:
            reverse = self.spatial_order == "native"
        for tensor in (image, mask):
            if tensor is not None and getattr(tensor, "pending_operations", []):
                raise ValueError("Materialize pending MONAI lazy operations before augmentation")
        image_array, image_spec = _to_volume_array(
            image,
            channel_dim=self.channel_dim,
            preserve_dtype=self.preserve_image_dtype,
            reverse=reverse,
        )
        mask_array, mask_spec = None, None
        if mask is not None:
            mask_array, mask_spec = _to_volume_array(
                mask,
                channel_dim=self.channel_dim,
                preserve_dtype=self.preserve_mask_dtype,
                reverse=reverse,
            )
        if affine is None:
            affine = getattr(image, "affine", None)
        if affine is not None:
            affine = _as_numpy(affine)[0] if _is_torch_tensor(affine) else np.asarray(affine)
            mask_affine = getattr(mask, "affine", None)
            if mask_affine is not None and not np.allclose(_as_numpy(mask_affine)[0], affine):
                raise ValueError("Image and label patient-space grids disagree")
            if reverse:
                affine = affine @ axis_permutation(image_array.ndim)
        spacing = () if spacing is None else tuple(float(s) for s in spacing)
        if reverse:
            spacing = spacing[::-1]
        target_values = {}
        for key, value in (targets or {}).items():
            arr = _as_numpy(value)[0] if _is_torch_tensor(value) else np.asarray(value)
            if reverse and key in ("keypoints", "bboxes"):
                ndim = image_array.ndim
                arr = (
                    arr[:, ::-1]
                    if key == "keypoints"
                    else np.concatenate([arr[:, :ndim][:, ::-1], arr[:, ndim:][:, ::-1]], axis=1)
                )
            target_values[key] = arr
        volume = MedVolume(
            image_array,
            mask_array,
            spacing,
            {} if metadata is None else dict(metadata),
            affine=affine,
            **target_values,
        )
        augmented = self.transform(volume)
        output_affine = augmented.affine
        if output_affine is not None and reverse:
            output_affine = output_affine @ axis_permutation(augmented.ndim)
        return (
            _restore_array(augmented.image, image_spec, output_affine),
            None
            if mask_spec is None or augmented.mask is None
            else _restore_array(augmented.mask, mask_spec, output_affine),
            augmented,
        )

    def _call_mapping(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        if self.image_key not in sample:
            raise KeyError(f"image_key={self.image_key!r} not found in sample")
        if self.require_mask and self.mask_key is not None and self.mask_key not in sample:
            raise KeyError(f"mask_key={self.mask_key!r} not found in sample")
        metadata = sample.get(self.metadata_key) if self.metadata_key else None
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata_key must point to a mapping")
        keys = ("keypoints", "keypoint_labels", "bboxes", "bbox_labels")
        targets = {key: sample[key] for key in keys if key in sample and sample[key] is not None}
        image, mask, augmented = self._augment(
            image=sample[self.image_key],
            mask=sample.get(self.mask_key) if self.mask_key else None,
            spacing=sample.get(self.spacing_key) if self.spacing_key else None,
            metadata=metadata,
            affine=sample.get(self.affine_key) if self.affine_key else None,
            targets=targets,
        )
        out = dict(sample)
        out[self.image_key] = image
        if self.mask_key is not None and self.mask_key in sample:
            out[self.mask_key] = mask
        reverse = self.spatial_order == "native"
        if self.spacing_key and self.spacing_key in sample:
            out[self.spacing_key] = augmented.spacing[::-1] if reverse else augmented.spacing
        if self.metadata_key and self.metadata_key in sample:
            out[self.metadata_key] = augmented.metadata
        if self.affine_key and augmented.affine is not None:
            out[self.affine_key] = (
                augmented.affine @ axis_permutation(augmented.ndim)
                if reverse
                else augmented.affine.copy()
            )
        for key in targets:
            value = getattr(augmented, key)
            if reverse and key in ("keypoints", "bboxes"):
                n = augmented.ndim
                value = (
                    value[:, ::-1]
                    if key == "keypoints"
                    else np.concatenate([value[:, :n][:, ::-1], value[:, n:][:, ::-1]], axis=1)
                )
            if _is_torch_tensor(sample[key]):
                import torch

                value = torch.as_tensor(value.copy(), device=sample[key].device)
            out[key] = value
        return out

    def __repr__(self) -> str:
        return f"{type(self).__name__}(transform={self.transform!r}, spatial_order={self.spatial_order!r})"


class TorchTransform(SampleTransform):
    """PyTorch dataset adapter using DHW/HW array order by default."""


class MonaiMapTransform(SampleTransform):
    """MONAI adapter using native spatial order and retaining current MetaTensor affine.

    Inversion history is explicitly cleared; MedAugmentX is not MONAI-invertible.
    Pass label_key=None for image-only samples, or spatial_order='array' for a
    custom MONAI pipeline that has already converted its spatial axes to DHW.
    """

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
        spatial_order: str = "native",
        affine_key: str | None = "affine",
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
            spatial_order=spatial_order,
            affine_key=affine_key,
            require_mask=label_key is not None,
        )


def _looks_like_torchio_image(value: Any) -> bool:
    return hasattr(value, "data") and hasattr(value, "set_data") and not _is_torch_tensor(value)


class TorchIOTransform(SampleTransform):
    """TorchIO adapter for one intensity image plus one optional registered label.

    Actual TorchIO data is (C, X, Y, Z); spatial axes and affine are converted
    together. Explicit keys must exist. Additional untransformed images are
    rejected to prevent partially augmented multimodal subjects.
    """

    def __init__(
        self,
        transform: Transform,
        *,
        image_key: str | None = None,
        label_key: str | None = "auto",
        channel_dim: ChannelDim = 0,
        preserve_image_dtype: bool = False,
        preserve_label_dtype: bool = True,
        copy: bool = True,
    ) -> None:
        super().__init__(
            transform,
            channel_dim=channel_dim,
            preserve_image_dtype=preserve_image_dtype,
            preserve_mask_dtype=preserve_label_dtype,
        )
        self.torchio_image_key, self.torchio_label_key, self.copy = image_key, label_key, copy

    def _apply_images(self, image: Any, label: Any | None) -> None:
        native = image.data.ndim == 4
        affine = getattr(image, "affine", None)
        if label is not None and affine is not None:
            label_affine = getattr(label, "affine", None)
            if label_affine is None or not np.allclose(label_affine, affine):
                raise ValueError("TorchIO image and label patient-space grids disagree")
        out_image, out_label, volume = self._augment(
            image=image.data,
            mask=None if label is None else label.data,
            spacing=getattr(image, "spacing", None),
            affine=affine,
            reverse=native,
        )
        image.set_data(out_image)
        if volume.affine is not None:
            image.affine = (
                volume.affine @ axis_permutation(volume.ndim) if native else volume.affine
            )
        if label is not None:
            label.set_data(out_label)
            if volume.affine is not None:
                label.affine = image.affine.copy()

    def __call__(self, sample: Any) -> Any:
        if _looks_like_torchio_image(sample):
            out = copy.deepcopy(sample) if self.copy else sample
            self._apply_images(out, None)
            return out
        if not isinstance(sample, Mapping) or not any(
            _looks_like_torchio_image(v) for v in sample.values()
        ):
            return super().__call__(sample)
        out = copy.deepcopy(sample) if self.copy else sample
        images = {key: value for key, value in out.items() if _looks_like_torchio_image(value)}
        labels = {
            key: value
            for key, value in images.items()
            if "label" in type(value).__name__.lower()
            or any(s in key.lower() for s in ("label", "mask", "seg"))
            or getattr(value, "type", None) == "label"
        }
        image_key = self.torchio_image_key
        if image_key is None:
            candidates = list(set(images) - set(labels))
            if len(candidates) != 1:
                raise KeyError("Could not infer a unique TorchIO intensity image key")
            image_key = candidates[0]
        if image_key not in images:
            raise KeyError(f"image_key={image_key!r} not found in TorchIO subject")
        label_key = self.torchio_label_key
        if label_key == "auto":
            if len(labels) > 1:
                raise KeyError("Could not infer a unique TorchIO label key")
            label_key = next(iter(labels), None)
        if label_key is not None and label_key not in images:
            raise KeyError(f"label_key={label_key!r} not found in TorchIO subject")
        selected = {image_key, label_key}
        if set(images) - selected:
            raise ValueError(
                "Multiple images require a joint multimodal policy; refusing partial augmentation"
            )
        self._apply_images(out[image_key], None if label_key is None else out[label_key])
        return out
