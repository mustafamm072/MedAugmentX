"""Keypoint and bounding-box tracking through a spatial pipeline (MedAugmentX 0.9.0).

A ``MedVolume`` can carry geometric targets — landmark *keypoints* and *bounding
boxes* — next to the image and mask. Every spatial transform warps those targets
in lockstep with the pixels, so detection and landmark annotations stay aligned
through the whole augmentation pipeline. Intensity and artifact transforms leave
them untouched.

Coordinates use **array-index order**: ``(z, y, x)`` for 3D and ``(y, x)`` for
2D — the same order used to index ``image``. Boxes are laid out as
``[min…, max…]`` (low corner then high corner).

Run with:

    python examples/keypoints_bboxes.py
"""
from __future__ import annotations

import numpy as np

from medaugmentx import Compose, MedVolume
from medaugmentx.transforms import AnatomicCrop, RandomAffine, RandomFlip, Resize


def _put_marker(image: np.ndarray, yx: tuple[int, int], value: float = 1.0) -> None:
    """Draw a small bright blob so we can eyeball where the anatomy moved."""
    y, x = yx
    image[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = value


def main() -> None:
    # A synthetic 2D image with one bright landmark and a box around it.
    image = np.zeros((256, 256), dtype=np.float32)
    landmark = (120, 80)
    _put_marker(image, landmark)

    vol = MedVolume(
        image=image,
        keypoints=np.array([[float(landmark[0]), float(landmark[1])]]),
        keypoint_labels=np.array(["landmark"]),
        bboxes=np.array([[105.0, 65.0, 135.0, 95.0]]),  # [y_min, x_min, y_max, x_max]
        bbox_labels=np.array(["lesion"]),
        metadata={"modality": "DX"},
    )
    print("Input:", vol)
    print("  keypoint:", vol.keypoints[0], "label:", vol.keypoint_labels[0])
    print("  bbox:    ", vol.bboxes[0], "label:", vol.bbox_labels[0])

    # A spatial pipeline: horizontal flip, a rotation+scale, then resample.
    pipeline = Compose(
        [
            RandomFlip(axes=("x",), p_per_axis=1.0),
            RandomAffine(rotation=(20.0, 20.0), scale=(1.1, 1.1), order=1, p=1.0),
            Resize(size=(512, 512)),
        ],
        seed=7,
    )
    out = pipeline(vol)

    # The keypoint still lands on the moved anatomy; verify against the pixel peak.
    peak = np.unravel_index(int(np.argmax(out.image)), out.image.shape)
    print("\nAfter flip + affine + resize:")
    print("  keypoint:", np.round(out.keypoints[0], 1), "label:", out.keypoint_labels[0])
    print("  pixel peak (for reference):", peak)
    print("  bbox:    ", np.round(out.bboxes[0], 1), "label:", out.bbox_labels[0])

    # A random crop can push targets off-frame. Transforms keep faithful (even
    # negative) coordinates; prune them explicitly when you need clean labels.
    cropped = AnatomicCrop(size=(96, 96), foreground_prob=0.0, seed=1)(out)
    print("\nAfter a random crop (before pruning):")
    print("  keypoints in frame?", cropped.num_keypoints, "point(s), coords:",
          np.round(cropped.keypoints[0], 1))

    pruned = cropped.remove_out_of_bounds_targets(min_visibility=0.25)
    print("After remove_out_of_bounds_targets(min_visibility=0.25):")
    print(f"  {pruned.num_keypoints} keypoint(s), {pruned.num_bboxes} box(es) retained")

    # Intensity/artifact transforms never move targets — only spatial ones do.
    print("\nTargets are only touched by spatial transforms; intensity/artifact",
          "transforms pass them through unchanged.")


if __name__ == "__main__":
    main()
