"""Local augmentation inspection; plotting remains an optional dependency."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from medaugmentx.core import MedVolume, Transform
from medaugmentx.reproducibility import _nodes
from medaugmentx.validation import Guard, VolumeValidator


def volume_summary(volume: MedVolume, reference: MedVolume | None = None) -> dict[str, Any]:
    """Describe intensities, labels and targets without identifiers or source paths."""
    finite = volume.image[np.isfinite(volume.image)]
    report = VolumeValidator(forbid_constant=False).validate(volume, reference)
    return {
        "shape": list(volume.shape),
        "dtype": str(volume.image.dtype),
        "spacing_mm": list(volume.spacing),
        "geometry_known": volume.affine is not None,
        "finite_fraction": float(np.isfinite(volume.image).mean()),
        "intensity": None
        if not finite.size
        else {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
        },
        "labels": []
        if volume.mask is None
        else [
            {
                "id": int(label),
                "voxels": int(count),
                "physical_measure": float(count * volume.voxel_volume),
            }
            for label, count in zip(*np.unique(volume.mask, return_counts=True))
        ],
        "keypoints": volume.num_keypoints,
        "bboxes": volume.num_bboxes,
        "issues": [
            {"check": issue.check, "severity": issue.severity, "message": issue.message}
            for issue in report.issues
        ],
    }


def preview_augmentation(
    transform: Transform, volume: MedVolume, path: str, *, samples: int = 6, seed: int = 0
) -> dict[str, Any]:
    """Save a local image/mask montage and return a machine-readable QA report.

    Requires the ``viz`` extra. Uses a copied policy, common image display limits,
    and middle array-axis slices for 3D. This is structural visual inspection,
    not a clinical plausibility test. The PNG contains image data; store it with
    the same access controls as the source images.
    """
    if not isinstance(samples, int) or not 1 <= samples <= 24:
        raise ValueError("samples must be an integer in [1, 24]")
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ImportError("Preview plotting requires 'medaugmentx[viz]'") from exc
    policy = copy.deepcopy(transform)
    policy.reseed(seed)
    outputs: list[MedVolume | None] = [volume]
    summaries = [volume_summary(volume)]
    for _ in range(samples):
        try:
            candidate = policy(volume.copy())
            outputs.append(candidate)
            summaries.append(volume_summary(candidate, volume))
        except (ValueError, FloatingPointError) as exc:
            outputs.append(None)
            summaries.append({"error": type(exc).__name__, "message": str(exc)})
    columns = min(4, samples + 1)
    rows = (samples + columns) // columns
    figure = Figure(figsize=(3.3 * columns, 3.6 * rows), constrained_layout=True)
    axes = figure.subplots(rows, columns, squeeze=False).ravel()
    finite = volume.image[np.isfinite(volume.image)]
    low, high = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
    if high <= low:
        high = low + 1
    for index, (out, summary) in enumerate(zip(outputs, summaries)):
        ax = axes[index]
        ax.set_axis_off()
        if out is None:
            ax.text(0.5, 0.5, summary["error"], color="red", ha="center")
            continue
        image = out.image[out.shape[0] // 2] if out.is_3d else out.image
        ax.imshow(image, cmap="gray", vmin=low, vmax=high)
        if out.mask is not None:
            mask = out.mask[out.shape[0] // 2] if out.is_3d else out.mask
            ax.imshow(
                np.ma.masked_where(mask == 0, mask),
                cmap="tab20",
                alpha=0.35,
                interpolation="nearest",
            )
        errors = sum(issue["severity"] == "error" for issue in summary["issues"])
        ax.set_title(
            ("Input" if index == 0 else f"Draw {index}") + f" | {errors} error(s)", fontsize=10
        )
    for ax in axes[len(outputs) :]:
        ax.set_axis_off()
    figure.savefig(path, dpi=130)
    guards = {
        node_path: node.stats for node_path, node in _nodes(policy) if isinstance(node, Guard)
    }
    return {
        "seed": seed,
        "samples": summaries,
        "guard_stats": guards,
        "scope": "Structural QA and visual preview; no anatomical or clinical validation",
    }
