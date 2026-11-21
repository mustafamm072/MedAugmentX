"""Nearest-neighbour kernels must move IDs without numeric conversion."""

import numpy as np
import pytest

from medaugmentx import MedVolume
from medaugmentx.transforms import CompressionVariation, ElasticDeform, RandomAffine, Resize


@pytest.mark.parametrize(
    "transform",
    [
        Resize((2, 9, 12), order=0, antialias=False),
        RandomAffine(rotation=(0, 0), translation=(0.2, 0.2), order=0, mode="constant"),
        ElasticDeform(alpha=2, sigma=1, order=0, mode="constant", seed=42),
        CompressionVariation(scale=0.6, order=0),
    ],
    ids=lambda transform: type(transform).__name__,
)
@pytest.mark.parametrize(
    "labels",
    [
        np.array([0, 1, 2**52 + 1, 2**52 + 3, 2**53 - 1], dtype=">u8"),
        np.array([0, 2**52 + 1, 2**53 - 1, 2**64 - 2, 2**64 - 1], dtype=np.uint64),
        np.array([0, -(2**52 + 1), 2**52 + 1, -(2**63), 2**63 - 1], dtype=np.int64),
        np.array([0, 1, -2, 3, 4], dtype=np.float16),
    ],
    ids=["wide-but-under-2-to-53", "uint64", "int64", "float16"],
)
def test_mask_sampling_preserves_exact_ids_and_background(transform, labels):
    codes = np.arange(4 * 6 * 8).reshape((4, 6, 8)) % (len(labels) - 1) + 1
    volume = MedVolume(codes.astype(np.float32), labels[codes])
    result = transform(volume)
    assert result.mask.dtype == labels.dtype
    np.testing.assert_array_equal(result.mask, labels[result.image.astype(np.intp)])
