"""Spacing normalization and RAS-mm annotation import using only NumPy/SciPy."""

import numpy as np

from medaugmentx import Compose, MedVolume
from medaugmentx.serialization import from_json, to_json
from medaugmentx.transforms import CenterCrop, Pad, ResampleToSpacing


def main():
    # Array axes are z, y, x; the affine maps them into RAS millimetres.
    affine = np.array([[0, 0, -0.8, 20], [0, 1.2, 0, -10], [2, 0, 0, 5], [0, 0, 0, 1]])
    mask = np.zeros((8, 24, 32), np.uint8)
    mask[2:6, 6:18, 8:24] = 3
    volume = MedVolume(mask.astype(np.float32), mask=mask, affine=affine)
    world_landmarks = np.array([[7.6, 3.8, 12]])
    volume = volume.replace(keypoints=volume.world_to_index(world_landmarks))

    policy = Compose(
        [
            ResampleToSpacing(1.0, max_voxels=1_000_000),
            Pad((16, 32, 32)),
            CenterCrop((16, 32, 32)),
        ]
    )
    restored = from_json(to_json(policy))
    output = restored(volume)
    np.testing.assert_allclose(output.index_to_world(output.keypoints), world_landmarks)
    assert output.shape == (16, 32, 32)
    np.testing.assert_allclose(output.spacing, (1, 1, 1))
    assert output.mask.dtype == mask.dtype
    print(f"Input:  {volume.shape}, spacing={volume.spacing}")
    print(f"Output: {output.shape}, spacing={output.spacing}")
    print("World landmarks preserved through resample, pad, and crop.")


if __name__ == "__main__":
    main()
