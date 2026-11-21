"""Spacing normalization, RAS-mm annotations, and segmentation training patches."""

import numpy as np

from medaugmentx import Compose, MedVolume, apply_for_sample
from medaugmentx.serialization import from_json, to_json
from medaugmentx.transforms import (
    CenterCrop,
    ClassBalancedCrop,
    Pad,
    ResampleToGrid,
    ResampleToSpacing,
)


def main():
    # Array axes are z, y, x; the affine maps them into RAS millimetres.
    affine = np.array([[0, 0, -0.8, 20], [0, 1.2, 0, -10], [2, 0, 0, 5], [0, 0, 0, 1]])
    mask = np.zeros((8, 24, 32), np.uint8)
    mask[2:6, 6:18, 8:24] = 3
    volume = MedVolume(mask.astype(np.float32), mask=mask, affine=affine)
    world_landmarks = np.array([[7.6, 3.8, 12]])
    volume = volume.replace(keypoints=volume.world_to_index(world_landmarks))

    preprocess = ResampleToSpacing(1.0, max_voxels=1_000_000)
    prepared = from_json(to_json(preprocess))(volume)
    fixed_grid = Compose([Pad((16, 32, 32)), CenterCrop((16, 32, 32))])
    output = fixed_grid(prepared)
    np.testing.assert_allclose(output.index_to_world(output.keypoints), world_landmarks)
    assert output.shape == (16, 32, 32)
    np.testing.assert_allclose(output.spacing, (1, 1, 1))
    assert output.mask.dtype == mask.dtype
    print(f"Input:  {volume.shape}, spacing={volume.spacing}")
    print(f"Output: {output.shape}, spacing={output.spacing}")
    print("World landmarks preserved through resample, pad, and crop.")

    training_policy = ClassBalancedCrop((18, 16, 16), foreground_prob=1, labels=(3,))
    patch = apply_for_sample(training_policy, prepared, seed=42, sample_id="phantom", epoch=0)
    assert patch.shape == (18, 16, 16)
    assert patch.mask is not None and (patch.mask == 3).any()
    np.testing.assert_allclose(patch.index_to_world(patch.keypoints), world_landmarks)
    print(f"Training patch: {patch.shape}, spacing={patch.spacing}, contains class 3.")

    native_grid = ResampleToGrid(volume.shape, affine)
    resampled_back = native_grid(prepared)
    assert resampled_back.shape == volume.shape
    np.testing.assert_allclose(resampled_back.affine, affine, atol=1e-10)
    np.testing.assert_allclose(
        resampled_back.index_to_world(resampled_back.keypoints), world_landmarks
    )
    print(f"Original grid restored: {resampled_back.shape}, spacing={resampled_back.spacing}.")


if __name__ == "__main__":
    main()
