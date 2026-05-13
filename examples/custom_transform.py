"""How to author your own transform and use it inside Compose.

The contract:
  1. Subclass Transform, override apply().
  2. Always sample from self.rng (never numpy.random) so deterministic seeding works.
  3. Override to_dict() so the transform can be serialised and reloaded.
  4. Register the class in REGISTRY if you want JSON/YAML round-trips.

Run with:

    python examples/custom_transform.py
"""
from __future__ import annotations

from typing import Any

import numpy as np

from medaugmentx import Compose, MedVolume
from medaugmentx.core import Transform
from medaugmentx.serialization import REGISTRY, from_json, to_json
from medaugmentx.transforms import GaussianNoise


class IntensityShift(Transform):
    """Add a uniform random offset to all voxels."""

    def __init__(self, max_shift: float = 0.05, p: float = 1.0, seed=None):
        super().__init__(p=p, seed=seed)
        self.max_shift = float(max_shift)

    def apply(self, volume: MedVolume) -> MedVolume:
        delta = float(self.rng.uniform(-self.max_shift, self.max_shift))
        return volume.replace(image=volume.image + delta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {"max_shift": self.max_shift, "p": self.p},
        }


# Register so to_json / from_json can reconstruct this transform.
REGISTRY["IntensityShift"] = IntensityShift


def main() -> None:
    image = np.full((16, 16), 0.5, dtype=np.float32)
    vol = MedVolume(image=image)

    pipeline = Compose([
        IntensityShift(max_shift=0.1),
        GaussianNoise(std=0.01),
    ], seed=7)

    out = pipeline(vol)
    print("Mean before:", float(vol.image.mean()))
    print("Mean after :", float(out.image.mean()))

    # Serialisation round-trip — works because IntensityShift is in REGISTRY.
    json_str = to_json(pipeline)
    pipeline2 = from_json(json_str)
    out2 = pipeline2(vol)
    print("\nRound-trip identical:", np.array_equal(out.image, out2.image))
    print("Serialised pipeline  :", json_str[:120], "...")


if __name__ == "__main__":
    main()
