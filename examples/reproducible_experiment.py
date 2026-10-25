"""Run from the repository root: python examples/reproducible_experiment.py.

Print a reproducible experiment record using only synthetic data. No input
images, patient metadata, filesystem paths, or network access are needed.
"""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version

import numpy as np

from medaugmentx import MedVolume
from medaugmentx.presets import mri_pipeline
from medaugmentx.serialization import from_json, to_json


def main() -> None:
    data_seed = 123
    pipeline_seed = 42
    volume = MedVolume(np.random.default_rng(data_seed).random((8, 32, 32)).astype(np.float32))
    pipeline = mri_pipeline(seed=pipeline_seed)
    policy = to_json(pipeline)
    result = pipeline(volume)
    replay = from_json(policy)(volume)
    np.testing.assert_array_equal(result.image, replay.image)
    canonical_policy = json.dumps(json.loads(policy), sort_keys=True, separators=(",", ":"))
    record = {
        "data": {"kind": "synthetic", "seed": data_seed, "shape": list(volume.shape)},
        "environment": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("medaugmentx", "numpy", "scipy")},
        },
        "pipeline": json.loads(policy),
        "pipeline_sha256": hashlib.sha256(canonical_policy.encode()).hexdigest(),
        "output_sha256": hashlib.sha256(result.image.tobytes()).hexdigest(),
        "replay_verified": True,
    }
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
