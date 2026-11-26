"""Studio state and operations, independent of the HTTP layer.

A :class:`StudioSession` owns one loaded volume and the policy being edited.
Policies travel as the canonical ``Transform.to_dict()`` form, so anything the
studio builds is exactly what :func:`medaugmentx.serialization.from_dict`,
``from_json`` and ``from_yaml`` load in training code.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
import warnings
from collections import Counter
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from medaugmentx import phantoms, presets
from medaugmentx.core.base import Transform
from medaugmentx.core.compose import Compose
from medaugmentx.core.tracing import ACTIVE_TRACE
from medaugmentx.core.volume import MedVolume
from medaugmentx.qa import volume_summary
from medaugmentx.reproducibility import _nodes, record_augmentation, seed_for_sample
from medaugmentx.serialization import (
    REGISTRY,
    _make_serialisable,
    from_dict,
    from_json,
    from_yaml,
    to_json,
    to_yaml,
)
from medaugmentx.studio import render
from medaugmentx.studio.catalog import NESTING, catalog, children_of
from medaugmentx.validation import VolumeValidator

PHANTOMS: dict[str, Callable[[], MedVolume]] = {
    "mri": phantoms.mri_phantom,
    "ct": phantoms.ct_phantom,
    "xray": phantoms.xray_phantom,
    "dbt": phantoms.dbt_phantom,
}
PRESETS: dict[str, Callable[..., Compose]] = {
    "mri": presets.mri_pipeline,
    "ct": presets.ct_pipeline,
    "dxr": presets.dxr_pipeline,
    "dbt": presets.dbt_pipeline,
}
MAX_GALLERY = 24
MAX_STRESS_DRAWS = 500
STRESS_TIME_BUDGET_S = 120.0
_CHANNEL_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
_RECOVERABLE = (ValueError, FloatingPointError, TypeError, KeyError, IndexError)
# Browsers parse JSON numbers as doubles; larger integers lose precision there.
JS_SAFE_INTEGER = 2**53 - 1


def _tidy_weights(node: Any) -> Any:
    """Drop uniform OneOf weights (equivalent to None) so edits keep them valid."""
    if isinstance(node, dict):
        params = node.get("params")
        if node.get("name") == "OneOf" and isinstance(params, dict):
            weights = params.get("weights")
            if isinstance(weights, list) and len(set(weights)) <= 1:
                params["weights"] = None
        for value in (params or {}).values() if isinstance(params, dict) else ():
            if isinstance(value, (dict, list)):
                _tidy_weights(value)
    elif isinstance(node, list):
        for item in node:
            _tidy_weights(item)
    return node


def _unsafe_integers(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_unsafe_integers(v) for v in value.values())
    if isinstance(value, list):
        return any(_unsafe_integers(v) for v in value)
    return isinstance(value, int) and not isinstance(value, bool) and abs(value) > JS_SAFE_INTEGER


class PolicyError(ValueError):
    """A policy that cannot be built, with the path of the failing step."""

    def __init__(self, message: str, path: list[int | str]):
        super().__init__(message)
        self.path = path


def _error_text(exc: BaseException) -> str:
    text = str(exc).strip() or type(exc).__name__
    if isinstance(exc, KeyError) and text.startswith(("'", '"')):
        text = text[1:-1]
    return f"{type(exc).__name__}: {text}"


def _demo_targets(volume: MedVolume) -> MedVolume:
    """Synthetic labels, landmarks and a box so phantoms exercise every overlay."""
    image = volume.image
    finite = image[np.isfinite(image)]
    low, high = np.percentile(finite, (55, 92))
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[image > low] = 1
    mask[image > high] = 2
    if not (mask == 2).any() or not (mask == 1).any():
        return volume.replace(mask=mask)
    keypoints = np.array([np.argwhere(mask == label).mean(axis=0) for label in (1, 2)])
    region = np.argwhere(mask == 2)
    box = np.concatenate([region.min(axis=0), region.max(axis=0)]).astype(np.float64)
    return volume.replace(
        mask=mask,
        keypoints=keypoints,
        keypoint_labels=np.array(["tissue", "bright"]),
        bboxes=box[None, :],
        bbox_labels=np.array(["bright"]),
    )


def _suggested_params(name: str, volume: MedVolume | None) -> dict[str, Any]:
    """Valid constructor arguments for transforms with required parameters."""
    shape = (64, 64) if volume is None else volume.shape
    spacing = (1.0,) * len(shape) if volume is None else volume.spacing
    smaller = [max(1, int(round(n * 0.75))) for n in shape]
    if name == "Compose":
        return {"transforms": []}
    if name in ("OneOf", "SomeOf"):
        # Both need a child; an empty Compose is an explicit "no change" option.
        return {"transforms": [Compose([])]}
    if name in ("Guard", "PerChannel"):
        return {"transform": Compose([])}
    if name in ("AnatomicCrop", "ClassBalancedCrop", "CenterCrop"):
        return {"size": smaller}
    if name == "Resize":
        return {"size": list(shape)}
    if name == "Pad":
        return {"size": [n + 8 for n in shape]}
    if name == "ResampleToSpacing":
        return {"spacing": [round(float(s) * 1.5, 4) for s in spacing]}
    if name == "ResampleToGrid":
        if volume is not None and volume.affine is not None:
            affine = volume.affine
        else:
            affine = np.eye(4)
            for axis, value in enumerate(spacing):
                affine[axis, axis] = value
        return {"shape": list(shape), "affine": affine.tolist()}
    return {}


class StudioSession:
    """State for one studio: a loaded volume, its source, and the edited policy.

    Args:
        volume: Optional in-memory volume to start with.
        policy: Optional transform or ``to_dict()`` mapping to start editing.
        source: Human-readable label for ``volume``.
    """

    def __init__(
        self,
        volume: MedVolume | None = None,
        policy: Transform | Mapping[str, Any] | None = None,
        *,
        source: str | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._uploads = tempfile.mkdtemp(prefix="medaugmentx-studio-")
        self._paths: dict[str, str] = {}
        self.volume: MedVolume | None = None
        self.source = ""
        self._version = 0
        self._cache: tuple[str, dict[str, Any]] | None = None
        if volume is not None:
            self.set_volume(volume, source or "in-memory volume")
        self.policy: dict[str, Any] = {"name": "Compose", "params": {"transforms": []}}
        if policy is not None:
            node = policy.to_dict() if isinstance(policy, Transform) else dict(policy)
            self.policy = _make_serialisable(node)
            self.compile(self.policy)

    # ------------------------------------------------------------------ data
    def close(self) -> None:
        """Delete uploaded files held for this session."""
        shutil.rmtree(self._uploads, ignore_errors=True)

    def set_volume(self, volume: MedVolume, source: str) -> None:
        if not isinstance(volume, MedVolume):
            raise TypeError("volume must be a MedVolume")
        with self._lock:
            self.volume = volume
            self.source = source
            self._touch()

    def _touch(self) -> None:
        self._version += 1
        self._cache = None

    def load_phantom(self, name: str, *, demo_targets: bool = True) -> dict[str, Any]:
        if name not in PHANTOMS:
            raise ValueError(f"Unknown phantom {name!r}; choose from {sorted(PHANTOMS)}")
        volume = PHANTOMS[name]()
        if demo_targets:
            volume = _demo_targets(volume)
        label = f"{name.upper()} phantom" + (" with synthetic demo labels" if demo_targets else "")
        self._paths = {}
        self.set_volume(volume, label)
        return self.state()

    def load_path(
        self,
        image: str,
        mask: str | None = None,
        *,
        resample_mask: bool = False,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load NIfTI (``.nii``/``.nii.gz``), NumPy (``.npy``) or DICOM data."""
        image = os.path.expanduser(str(image).strip())
        mask = None if not mask else os.path.expanduser(str(mask).strip())
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image path does not exist: {image}")
        if mask is not None and not os.path.exists(mask):
            raise FileNotFoundError(f"Mask path does not exist: {mask}")
        lower = image.lower()
        if lower.endswith((".nii", ".nii.gz")):
            from medaugmentx.io import load_nifti

            volume = load_nifti(image, mask_path=mask, resample_mask=resample_mask)
        elif lower.endswith(".npy"):
            array = np.load(image, allow_pickle=False)
            labels = None if mask is None else np.load(mask, allow_pickle=False)
            volume = MedVolume(array, labels)
        else:
            if mask is not None:
                raise ValueError("Masks are supported for NIfTI and NumPy images")
            from medaugmentx.io import load_dicom_series

            volume = load_dicom_series(image, strict=strict)
        self._paths = {"image": image, **({"mask": mask} if mask else {})}
        self.set_volume(volume, os.path.basename(image.rstrip(os.sep)) or image)
        self._prune_uploads()
        return self.state()

    def attach_mask(self, path: str, *, resample_mask: bool = False) -> dict[str, Any]:
        """Attach a label map to the loaded image, keeping its channels and targets.

        A NIfTI mask for a NIfTI image opened from disk is checked (and optionally
        resampled) exactly as ``load_nifti(image, mask_path=...)`` would. Other
        masks must already share the image grid.
        """
        path = os.path.expanduser(str(path).strip())
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mask path does not exist: {path}")
        with self._lock:
            volume = self._require_volume()
            lower = path.lower()
            if lower.endswith(".npy"):
                labels = np.load(path, allow_pickle=False)
            elif lower.endswith((".nii", ".nii.gz")):
                from medaugmentx.core.resampling import grids_match
                from medaugmentx.io import load_nifti

                image_path = self._paths.get("image", "")
                if image_path.lower().endswith((".nii", ".nii.gz")):
                    reloaded = load_nifti(image_path, mask_path=path, resample_mask=resample_mask)
                    labels = reloaded.mask
                else:
                    labelled = load_nifti(path, dtype=None)
                    if labelled.shape != volume.shape or (
                        volume.affine is not None
                        and labelled.affine is not None
                        and not grids_match(volume.affine, labelled.affine, volume.shape)
                    ):
                        raise ValueError(
                            "Mask must share the image grid; open image and mask from disk "
                            "to resample a registered mask"
                        )
                    labels = labelled.image
            else:
                raise ValueError("Masks must be .nii, .nii.gz or .npy files")
            if labels is None or labels.shape != volume.shape:
                shape = None if labels is None else labels.shape
                raise ValueError(f"Mask shape {shape} does not match image shape {volume.shape}")
            self.volume = volume.replace(mask=labels)
            self._paths["mask"] = path
            self._touch()
            self._prune_uploads()
        return self.state()

    def _prune_uploads(self) -> None:
        """Delete uploaded files that no longer back the loaded data."""
        keep = {os.path.abspath(p) for p in self._paths.values()}
        try:
            names = os.listdir(self._uploads)
        except OSError:
            return
        for name in names:
            candidate = os.path.join(self._uploads, name)
            if os.path.abspath(candidate) not in keep:
                try:
                    os.remove(candidate)
                except OSError:
                    pass

    def upload_target(self, role: str, filename: str) -> str:
        """Private path for an upload, validating its role and file type."""
        if role not in ("image", "mask", "channel"):
            raise ValueError("Upload role must be image, mask or channel")
        lower = (os.path.basename(str(filename)) or "upload").lower()
        if not lower.endswith((".nii", ".nii.gz", ".npy")):
            raise ValueError("Uploads must be .nii, .nii.gz or .npy files")
        suffix = ".nii.gz" if lower.endswith(".nii.gz") else os.path.splitext(lower)[1]
        return os.path.join(self._uploads, f"{role}-{time.monotonic_ns()}{suffix}")

    def store_upload(self, role: str, filename: str, data: bytes) -> str:
        """Save uploaded bytes under the session's private folder."""
        target = self.upload_target(role, filename)
        with open(target, "wb") as handle:
            handle.write(data)
        return target

    def add_channel(self, name: str, path: str) -> dict[str, Any]:
        """Attach a registered image as a named channel of the loaded volume."""
        with self._lock:
            if self.volume is None:
                raise ValueError("Load an image before adding channels")
            if not _CHANNEL_NAME.match(str(name)):
                raise ValueError("Channel names use 1-64 letters, digits, '.', '_' or '-'")
            path = os.path.expanduser(str(path).strip())
            if path.lower().endswith(".npy"):
                array = np.load(path, allow_pickle=False)
                if name in self.volume.channels:
                    raise ValueError(f"Channel {name!r} already exists")
                channels = {**self.volume.channels, name: array}
                volume = self.volume.replace(channels=channels)
            elif path.lower().endswith((".nii", ".nii.gz")):
                from medaugmentx.io import load_nifti

                volume = self.volume.with_channels({name: load_nifti(path)})
            else:
                raise ValueError("Channels must be .nii, .nii.gz or .npy files")
            self.volume = volume
            self._touch()
            self._prune_uploads()
        return self.state()

    def remove_channel(self, name: str) -> dict[str, Any]:
        with self._lock:
            if self.volume is None or name not in self.volume.channels:
                raise ValueError(f"No channel named {name!r}")
            channels = {k: v for k, v in self.volume.channels.items() if k != name}
            self.volume = self.volume.replace(channels=channels)
            self._touch()
        return self.state()

    def _require_volume(self) -> MedVolume:
        if self.volume is None:
            raise ValueError("Load an image or phantom first")
        return self.volume

    def state(self) -> dict[str, Any]:
        """Loaded-data summary, axis names, channels and the current policy."""
        with self._lock:
            volume = self.volume
            info: dict[str, Any] = {"loaded": volume is not None, "policy": self.policy}
            if volume is None:
                return info
            summary = volume_summary(volume)
            info.update(
                {
                    "source": self.source,
                    "modality": volume.modality,
                    "ndim": volume.ndim,
                    "axes": list(render.axis_names(volume.ndim)),
                    "channels": list(volume.channels),
                    "has_mask": volume.mask is not None,
                    "summary": summary,
                }
            )
            return info

    # --------------------------------------------------------------- policy
    def compile(self, node: Any) -> Transform:
        """Build a transform, reporting the deepest failing step on error."""
        try:
            return from_dict(copy.deepcopy(node))
        except Exception as exc:  # noqa: BLE001 - reported to the editor
            path, message = self._locate(node, [], exc)
            raise PolicyError(message, path) from exc

    def _locate(
        self, node: Any, path: list[int | str], fallback: BaseException
    ) -> tuple[list[int | str], str]:
        if isinstance(node, dict):
            for key, index, child in children_of(node):
                child_path = path + ([index] if index is not None else [key])
                try:
                    from_dict(copy.deepcopy(child))
                except Exception as exc:  # noqa: BLE001
                    return self._locate(child, child_path, exc)
        return path, _error_text(fallback)

    def set_policy(self, node: Any) -> dict[str, Any]:
        """Store the edited policy and report whether it builds."""
        if not isinstance(node, dict):
            raise ValueError("Policy must be an object with name and params")
        with self._lock:
            self.policy = node
        try:
            transform = self.compile(node)
        except PolicyError as exc:
            return {"ok": False, "error": str(exc), "path": exc.path}
        from medaugmentx.inspection import pipeline_summary

        return {"ok": True, "summary": pipeline_summary(transform)}

    def current_policy(self) -> Transform:
        """The edited policy as a fresh transform (raises if it does not build)."""
        with self._lock:
            return self.compile(self.policy)

    def new_step(self, name: str) -> dict[str, Any]:
        """Canonical node for a newly added step, with volume-aware defaults."""
        if name not in REGISTRY:
            raise ValueError(f"Unknown transform {name!r}")
        params = _suggested_params(name, self.volume)
        transform = REGISTRY[name](**params)
        return _tidy_weights(_make_serialisable(transform.to_dict()))

    def preset(self, name: str, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if name not in PRESETS:
            raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}")
        factory = PRESETS[name]
        allowed = {
            key
            for key, parameter in inspect.signature(factory).parameters.items()
            if parameter.kind == parameter.KEYWORD_ONLY
        }
        kwargs = {key: bool(value) for key, value in (options or {}).items() if key in allowed}
        return _tidy_weights(_make_serialisable(factory(**kwargs).to_dict()))

    # -------------------------------------------------------------- running
    def _run(
        self,
        node: Any,
        *,
        volume: MedVolume,
        seed: int,
        sample_id: str | int,
        epoch: int,
        index: int,
    ) -> dict[str, Any]:
        transform = self.compile(node)
        sample_seed = seed_for_sample(seed, sample_id, epoch=epoch, augmentation_index=index)
        transform.reseed(sample_seed)
        names = {id(n): path for path, n in _nodes(transform)}
        events: list[tuple[float, str, str, str]] = []

        def tracer(step: Transform, action: str) -> None:
            events.append(
                (time.perf_counter(), names.get(id(step), "?"), type(step).__name__, action)
            )

        start = time.perf_counter()
        token = ACTIVE_TRACE.set(tracer)
        output: MedVolume | None = None
        error = None
        try:
            output = transform(volume.copy())
        except _RECOVERABLE as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            ACTIVE_TRACE.reset(token)
        end = time.perf_counter()
        stamps = [stamp for stamp, *_ in events] + [end]
        trace = [
            {
                "path": path,
                "name": name,
                "action": action,
                "ms": round(1000 * (stamps[i + 1] - stamp), 3) if action == "applied" else 0.0,
            }
            for i, (stamp, path, name, action) in enumerate(events)
        ]
        return {
            "output": output,
            "error": error,
            "trace": trace,
            "elapsed_ms": round(1000 * (end - start), 3),
            "sample_seed": str(sample_seed),
        }

    def _cached_run(self, node: Any, draw: Mapping[str, Any]) -> dict[str, Any]:
        """Reuse the last draw when only the displayed slice or window changes."""
        key = json.dumps([node, dict(draw), self._version], sort_keys=True, default=str)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        result = self._run(node, volume=self._require_volume(), **draw)
        self._cache = (key, result)
        return result

    @staticmethod
    def _draw_options(request: Mapping[str, Any]) -> dict[str, Any]:
        def integer(key: str, default: int) -> int:
            value = request.get(key, default)
            invalid = ValueError(f"{key} must be a non-negative integer")
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise invalid
            if isinstance(value, float) and not (math.isfinite(value) and value.is_integer()):
                raise invalid
            if isinstance(value, str) and not value.strip().isdigit():
                raise invalid
            number = int(value)
            if number < 0:
                raise invalid
            return number

        sample_id = request.get("sample_id", "studio")
        if isinstance(sample_id, str):
            sample_id = sample_id.strip() or "studio"
            # The browser sends digit IDs as text so large integers stay exact.
            if request.get("numeric_sample_id") and sample_id.isdigit():
                sample_id = int(sample_id)
        elif isinstance(sample_id, bool) or not isinstance(sample_id, int):
            raise ValueError("sample_id must be a string or integer")
        return {
            "seed": integer("seed", 0),
            "sample_id": sample_id,
            "epoch": integer("epoch", 0),
            "index": integer("index", 0),
        }

    def _view_options(self, request: Mapping[str, Any], volume: MedVolume) -> dict[str, Any]:
        axes = render.axis_names(volume.ndim)
        axis_name = request.get("axis", axes[0])
        if axis_name not in axes:
            raise ValueError(f"axis must be one of {list(axes)}")
        channel = request.get("channel") or None
        if channel is not None and channel not in volume.channels:
            raise ValueError(f"Unknown channel {channel!r}")
        alpha = float(request.get("mask_alpha", 0.4))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("mask_alpha must be in [0, 1]")
        return {
            "axis": axes.index(axis_name) if volume.ndim == 3 else 0,
            "position": float(request.get("position", 0.5)),
            "channel": channel,
            "mask_alpha": alpha,
            "window_mode": request.get("window_mode", "input"),
            "window": request.get("window"),
        }

    @staticmethod
    def _image_of(volume: MedVolume, channel: str | None) -> np.ndarray:
        return volume.image if channel is None else volume.channels[channel]

    def _window_for(
        self, mode: str, manual: Any, image: np.ndarray, reference: np.ndarray
    ) -> tuple[float, float]:
        if mode == "manual":
            try:
                low, high = (float(v) for v in manual)
            except (TypeError, ValueError) as exc:
                raise ValueError("Manual window needs numeric [low, high]") from exc
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError("Manual window needs finite low < high")
            return low, high
        if mode == "auto":
            return render.display_window(image)
        if mode != "input":
            raise ValueError("window_mode must be input, auto or manual")
        return render.display_window(reference)

    def preview(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Run one seeded draw and render matching input/output slices."""
        with self._lock:
            volume = self._require_volume()
            draw = self._draw_options(request)
            view = self._view_options(request, volume)
            result = self._cached_run(request.get("policy", self.policy), draw)
            channel = view["channel"]
            reference = self._image_of(volume, channel)
            in_window = self._window_for(view["window_mode"], view["window"], reference, reference)
            source = render.render_view(
                volume,
                axis=view["axis"],
                position=view["position"],
                channel=channel,
                window=in_window,
                mask_alpha=view["mask_alpha"],
            )
            output = result["output"]
            augmented = None
            diff = None
            issues: list[dict[str, str]] = []
            out_summary = None
            out_window = in_window
            if output is not None:
                out_window = self._window_for(
                    view["window_mode"], view["window"], self._image_of(output, channel), reference
                )
                augmented = render.render_view(
                    output,
                    axis=view["axis"],
                    position=view["position"],
                    channel=channel,
                    window=out_window,
                    mask_alpha=view["mask_alpha"],
                )
                if output.shape == volume.shape:
                    image = render.difference_image(source["plain"], augmented["plain"])
                    diff = None if image is None else render.png_data_url(image)
                report = VolumeValidator().validate(output, volume)
                issues = [
                    {"check": i.check, "severity": i.severity, "message": i.message}
                    for i in report.issues
                ]
                out_summary = _brief(output)
            for view_result in (source, augmented):
                if view_result is not None:
                    view_result.pop("plain")
            return {
                **draw,
                "sample_id": draw["sample_id"],
                "sample_seed": result["sample_seed"],
                "elapsed_ms": result["elapsed_ms"],
                "error": result["error"],
                "trace": result["trace"],
                "issues": issues,
                "input": source,
                "output": augmented,
                "diff": diff,
                "window": {"input": list(in_window), "output": list(out_window)},
                "summary": {"input": _brief(volume), "output": out_summary},
            }

    def _snapshot(self, request: Mapping[str, Any]) -> tuple[MedVolume, Any]:
        """Volume and policy for a long run, taken under the lock.

        Long runs then proceed without the lock so edits, state and previews stay
        responsive. Volumes are replaced, never mutated, so the snapshot is stable.
        """
        with self._lock:
            volume = self._require_volume()
            node = copy.deepcopy(request.get("policy", self.policy))
        self.compile(node)
        return volume, node

    def gallery(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Render consecutive augmentation indices for a quick variability scan."""
        draw = self._draw_options(request)
        count = request.get("count", 12)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_GALLERY:
            raise ValueError(f"count must be an integer between 1 and {MAX_GALLERY}")
        volume, node = self._snapshot(request)
        view = self._view_options(request, volume)
        channel = view["channel"]
        reference = self._image_of(volume, channel)
        items = []
        for offset in range(count):
            index = draw["index"] + offset
            result = self._run(node, volume=volume, **{**draw, "index": index})
            output = result["output"]
            item: dict[str, Any] = {
                "index": index,
                "elapsed_ms": result["elapsed_ms"],
                "error": result["error"],
                "applied": [e["name"] for e in result["trace"] if e["action"] == "applied"],
            }
            if output is not None:
                window = self._window_for(
                    view["window_mode"], view["window"], self._image_of(output, channel), reference
                )
                rendered = render.render_view(
                    output,
                    axis=view["axis"],
                    position=view["position"],
                    channel=channel,
                    window=window,
                    mask_alpha=view["mask_alpha"],
                )
                rendered.pop("plain")
                report = VolumeValidator().validate(output, volume)
                item.update(
                    {
                        "view": rendered,
                        "errors": len(report.errors),
                        "warnings": len(report.warnings),
                    }
                )
            items.append(item)
        return {**draw, "items": items}

    def stress(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Run many draws and aggregate validation, failures, timing and branches."""
        draw = self._draw_options(request)
        draws = request.get("draws", 50)
        if (
            isinstance(draws, bool)
            or not isinstance(draws, int)
            or not 1 <= draws <= MAX_STRESS_DRAWS
        ):
            raise ValueError(f"draws must be an integer between 1 and {MAX_STRESS_DRAWS}")
        config = request.get("validator")
        validator = VolumeValidator() if not config else VolumeValidator.from_dict(config)
        volume, node = self._snapshot(request)
        timings: list[float] = []
        checks: Counter[str] = Counter()
        exceptions: Counter[str] = Counter()
        shapes: Counter[str] = Counter()
        applied: Counter[str] = Counter()
        labels: dict[str, str] = {}
        rejected = completed = 0
        started = time.perf_counter()
        for offset in range(draws):
            if time.perf_counter() - started > STRESS_TIME_BUDGET_S:
                break
            result = self._run(node, volume=volume, **{**draw, "index": draw["index"] + offset})
            completed += 1
            timings.append(result["elapsed_ms"])
            # Count each step once per draw: PerChannel and Guard retries can
            # apply the same child several times within one draw.
            ran = set()
            for event in result["trace"]:
                if event["action"] == "applied":
                    ran.add(event["path"])
                labels[event["path"]] = event["name"]
            applied.update(ran)
            if result["error"] is not None:
                exceptions[f"{result['error']['type']}: {result['error']['message']}"] += 1
                continue
            output = result["output"]
            shapes["×".join(str(n) for n in output.shape)] += 1
            report = validator.validate(output, volume)
            if not report.ok:
                rejected += 1
            for issue in report.issues:
                checks[f"{issue.severity}:{issue.check}"] += 1
        times = np.asarray(timings) if timings else np.zeros(1)
        # ".inner" nodes are private helpers of a wrapper, not editor steps.
        branch_paths = [path for path, _ in _nodes(self.compile(node)) if ".inner" not in path]
        return {
            **draw,
            "requested": draws,
            "completed": completed,
            "truncated": completed < draws,
            "exceptions": sum(exceptions.values()),
            "rejected": rejected,
            "passed": completed - rejected - sum(exceptions.values()),
            "issues": [
                {"severity": key.split(":", 1)[0], "check": key.split(":", 1)[1], "count": n}
                for key, n in checks.most_common()
            ],
            "exception_messages": [
                {"message": key, "count": n} for key, n in exceptions.most_common(8)
            ],
            "shapes": [{"shape": key, "count": n} for key, n in shapes.most_common(8)],
            "applied": [
                {
                    "path": path,
                    "name": labels.get(path, ""),
                    "rate": applied[path] / completed if completed else 0.0,
                }
                for path in branch_paths
                if path != "root"
            ],
            "timing_ms": {
                "mean": float(times.mean()),
                "median": float(np.median(times)),
                "p95": float(np.percentile(times, 95)),
                "max": float(times.max()),
            },
            "validator": validator.to_dict(),
        }

    def record(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Exact replay record for one draw (see ``replay_augmentation``)."""
        with self._lock:
            volume = self._require_volume()
            draw = self._draw_options(request)
            transform = self.compile(request.get("policy", self.policy))
            transform.reseed(
                seed_for_sample(
                    draw["seed"],
                    draw["sample_id"],
                    epoch=draw["epoch"],
                    augmentation_index=draw["index"],
                )
            )
            _, record = record_augmentation(transform, volume)
            return _make_serialisable(record)

    # --------------------------------------------------------- import/export
    def export(self, node: Any, fmt: str) -> dict[str, Any]:
        transform = self.compile(node)
        if fmt == "json":
            return {"format": "json", "filename": "policy.json", "text": to_json(transform)}
        if fmt == "yaml":
            try:
                text = to_yaml(transform)
            except ImportError as exc:
                raise ValueError("YAML export requires PyYAML: pip install pyyaml") from exc
            return {"format": "yaml", "filename": "policy.yaml", "text": text}
        if fmt == "python":
            return {"format": "python", "filename": "policy.py", "text": python_snippet(node)}
        raise ValueError("format must be json, yaml or python")

    def import_text(self, text: str) -> dict[str, Any]:
        """Parse a JSON or YAML policy document into the canonical editor form."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Paste or choose a JSON or YAML policy document")
        notices: list[str] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                json.loads(text)
                transform = from_json(text)
            except json.JSONDecodeError:
                try:
                    import yaml
                except ImportError as exc:
                    raise ValueError("YAML import requires PyYAML: pip install pyyaml") from exc
                try:
                    transform = from_yaml(text)
                except yaml.YAMLError as exc:
                    raise ValueError(f"Not a valid JSON or YAML policy: {exc}") from exc
        notices.extend(str(w.message) for w in caught)
        node = _tidy_weights(_make_serialisable(transform.to_dict()))
        if _unsafe_integers(node):
            notices.append(
                "This policy contains integers above 2**53 (for example 64-bit seeds); "
                "browsers cannot represent them exactly, so set them in Python instead."
            )
        return {"policy": node, "notices": notices}


def _brief(volume: MedVolume) -> dict[str, Any]:
    finite = volume.image[np.isfinite(volume.image)]
    return {
        "shape": list(volume.shape),
        "spacing": [round(float(s), 6) for s in volume.spacing],
        "dtype": str(volume.image.dtype),
        "intensity": None
        if not finite.size
        else {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
        },
        "labels": [] if volume.mask is None else [int(v) for v in np.unique(volume.mask) if v],
        "keypoints": volume.num_keypoints,
        "bboxes": volume.num_bboxes,
        "channels": list(volume.channels),
    }


# ---------------------------------------------------------------- codegen
_TOP_LEVEL = {"Compose", "OneOf", "SomeOf", "PerChannel", "Guard"}
_LINE = 88


def _fits(indent: str, text: str) -> bool:
    return "\n" not in text and len(indent) + len(text) <= _LINE


def _call(name: str, args: list[str], indent: str) -> str:
    flat = f"{name}({', '.join(args)})"
    if _fits(indent, flat):
        return flat
    inner = indent + "    "
    return f"{name}(\n" + "".join(f"{inner}{arg},\n" for arg in args) + f"{indent})"


def _import_line(module: str, names: list[str]) -> str:
    flat = f"from {module} import {', '.join(names)}"
    if len(flat) <= _LINE:
        return flat
    return f"from {module} import (\n" + "".join(f"    {name},\n" for name in names) + ")"


def python_snippet(node: Mapping[str, Any]) -> str:
    """Readable Python that constructs the same policy, omitting defaults."""
    import medaugmentx.transforms as builtin

    used: set[str] = set()

    def expression(item: Mapping[str, Any], indent: str) -> str:
        name = str(item["name"])
        used.add(name)
        signature = inspect.signature(REGISTRY[name])
        params = dict(item.get("params") or {})
        inner = indent + "    "
        args: list[str] = []
        key = NESTING.get(name)
        if key == "transforms":
            children = [expression(c, inner + "    ") for c in params.pop("transforms", []) or []]
            flat = "[" + ", ".join(children) + "]"
            if _fits(inner, flat):
                args.append(flat)
            else:
                body = "".join(f"{inner}    {child},\n" for child in children)
                args.append(f"[\n{body}{inner}]")
        elif key == "transform":
            args.append(expression(params.pop("transform"), inner))
        weights = params.get("weights")
        if name == "OneOf" and isinstance(weights, list) and len(set(weights)) <= 1:
            params.pop("weights")
        if name == "Guard" and isinstance(params.get("validator"), dict):
            defaults = _make_serialisable(VolumeValidator().to_dict())
            changed = {k: v for k, v in params["validator"].items() if defaults.get(k) != v}
            if changed:
                params["validator"] = changed
            else:
                params.pop("validator")
        for param, value in params.items():
            default = signature.parameters.get(param)
            if default is not None and default.default is not inspect.Parameter.empty:
                if _make_serialisable(default.default) == value:
                    continue
            args.append(f"{param}={value!r}")
        return _call(name, args, indent)

    body = expression(node, "")
    core = sorted(used & _TOP_LEVEL)
    leaves = sorted(name for name in used - _TOP_LEVEL if name in builtin.__all__)
    custom = sorted(used - _TOP_LEVEL - set(leaves))
    lines = ['"""Augmentation policy exported from MedAugmentX Policy Studio."""', ""]
    if core:
        lines.append(_import_line("medaugmentx", core))
    if leaves:
        lines.append(_import_line("medaugmentx.transforms", leaves))
    for name in custom:
        lines.append(f"# Import and register your custom transform {name} before use.")
    lines += ["", f"policy = {body}", ""]
    return "\n".join(lines)


__all__ = ["PHANTOMS", "PRESETS", "PolicyError", "StudioSession", "catalog", "python_snippet"]
