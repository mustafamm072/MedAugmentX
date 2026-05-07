"""How to author your own transform and use it inside Compose.

The contract: subclass Transform, override apply(), and always sample from
self.rng (never numpy.random) so deterministic seeding works.
"""
from __future__ import annotations

import numpy as np

from medaugment import Compose, MedVolume
from medaugment.core import Transform
from medaugment.transforms import GaussianNoise


class IntensityShift(Transform):
    """Add a uniform random offset to all voxels."""

    def __init__(self, max_shift: float = 0.05, p: float = 1.0, seed=None):
        super().__init__(p=p, seed=seed)
        self.max_shift = float(max_shift)

    def apply(self, volume: MedVolume) -> MedVolume:
        delta = float(self.rng.uniform(-self.max_shift, self.max_shift))
        return volume.replace(image=volume.image + delta)


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


if __name__ == "__main__":
    main()
