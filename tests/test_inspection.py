import numpy as np
import pytest

from medaugmentx import Compose, OneOf, iter_pipeline, pipeline_summary
from medaugmentx.core import Transform
from medaugmentx.inspection import PipelineStep
from medaugmentx.transforms import GaussianNoise, RandomFlip, RicianNoise


class ArrayParamTransform(Transform):
    def __init__(self):
        super().__init__()
        self.reference = np.zeros((3, 4), dtype=np.float32)

    def apply(self, volume):
        return volume


def test_iter_pipeline_yields_depth_first_steps():
    pipeline = Compose(
        [
            RandomFlip(axes=("x",), p=0.5),
            OneOf([GaussianNoise(std=0.01), RicianNoise(std=0.02)], weights=[0.25, 0.75]),
        ],
        seed=42,
    )

    steps = list(iter_pipeline(pipeline))

    assert [step.name for step in steps] == [
        "Compose",
        "RandomFlip",
        "OneOf",
        "GaussianNoise",
        "RicianNoise",
    ]
    assert [step.path for step in steps] == [(), (0,), (1,), (1, 0), (1, 1)]
    assert all(isinstance(step, PipelineStep) for step in steps)
    assert steps[2].params["weights"] == [0.25, 0.75]
    assert "transforms" not in steps[2].params


def test_pipeline_summary_formats_nested_pipeline():
    pipeline = Compose(
        [
            RandomFlip(axes=("x",), p=0.5),
            OneOf([GaussianNoise(std=0.01), RicianNoise(std=0.02)], p=0.3),
        ],
        seed=42,
    )

    summary = pipeline_summary(pipeline)

    assert summary.splitlines() == [
        "Compose(p=1.0, seed=42)",
        "  0 RandomFlip(axes=['x'], p_per_axis=0.5, p=0.5)",
        "  1 OneOf(weights=[0.5, 0.5], p=0.3, seed=None)",
        "    1.0 GaussianNoise(std=0.01, relative=False, clip=None, p=1.0)",
        "    1.1 RicianNoise(std=0.02, clip=None, p=1.0)",
    ]


def test_pipeline_summary_compacts_numpy_array_params():
    summary = pipeline_summary(ArrayParamTransform())

    assert summary.startswith("ArrayParamTransform(")
    assert "p=1.0" in summary
    assert "reference=ndarray(shape=(3, 4), dtype=float32)" in summary


def test_pipeline_summary_truncates_long_values():
    summary = pipeline_summary(RandomFlip(axes=("x", "y", "z")), max_value_length=8)

    assert "axes=['x',..." in summary


def test_iter_pipeline_rejects_non_transform():
    with pytest.raises(TypeError, match="Transform"):
        list(iter_pipeline(object()))  # type: ignore[arg-type]


def test_pipeline_summary_validates_max_value_length():
    with pytest.raises(ValueError, match="at least 8"):
        pipeline_summary(RandomFlip(), max_value_length=7)
