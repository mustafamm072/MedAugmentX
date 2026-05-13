# API Examples

Working code for common use-cases. Every snippet is self-contained — copy,
paste, run.

---

## 1. Build a `MedVolume` from a NumPy array

```python
import numpy as np
from medaugment import MedVolume

vol = MedVolume(
    image=np.random.rand(80, 256, 256).astype(np.float32),  # (D, H, W)
    mask=np.zeros((80, 256, 256), dtype=np.uint8),
    spacing=(1.0, 0.7, 0.7),                                 # mm (z, y, x)
    metadata={"modality": "MR", "patient_id": "anon-001"},
)
print(vol)
# MedVolume(image=shape=(80, 256, 256), dtype=float32, mask=shape=(80, 256, 256), ...)
```

---

## 2. Apply a single transform

```python
from medaugmentx.transforms import RandomFlip

augmented = RandomFlip(axes=("x",), p_per_axis=1.0, seed=0)(vol)
```

---

## 3. Compose a pipeline (deterministic seeding)

```python
from medaugment import Compose, OneOf
from medaugmentx.transforms import (
    RandomAffine, ElasticDeform,
    RicianNoise, GaussianNoise,
    GammaCorrection,
)

pipeline = Compose([
    RandomAffine(rotation=15, scale=(0.9, 1.1), p=0.7),
    ElasticDeform(alpha=(120, 120, 10), sigma=(10, 10, 3), p=0.5),
    OneOf([
        RicianNoise(std=0.02),
        GaussianNoise(std=0.015),
    ], p=0.6),
    GammaCorrection(gamma=(0.85, 1.15), p=0.5),
], seed=42)

out = pipeline(vol)
```

`Compose(..., seed=42)` is bit-deterministic: the same seed yields the same
output on every run, on every machine.

---

## 4. MRI pipeline with Phase 2 transforms

```python
from medaugment import Compose, OneOf
from medaugmentx.transforms import (
    RandomAffine, ElasticDeform, RandomFlip,
    BiasField, RicianNoise, GammaCorrection,
    GhostingArtifact, KSpaceDropout,
)

mri_pipeline = Compose([
    RandomFlip(axes=("x",), p_per_axis=0.5, p=0.5),
    RandomAffine(rotation=10, scale=(0.9, 1.1), translation=(-0.05, 0.05), p=0.7),
    ElasticDeform(alpha=30.0, sigma=4.0, p=0.5),
    BiasField(alpha=0.3, p=0.7),                      # MRI coil inhomogeneity
    RicianNoise(std=(0.005, 0.02), p=0.5),
    GammaCorrection(gamma=(0.85, 1.15), p=0.5),
    OneOf([
        GhostingArtifact(ghost_intensity=(0.05, 0.12)),  # phase-encode ghosting
        KSpaceDropout(dropout_fraction=(0.01, 0.04)),    # k-space line dropout
    ], p=0.3),
], seed=0)
```

Or use the pre-built preset:

```python
from medaugmentx.presets import mri_pipeline

pipeline = mri_pipeline(seed=0)
augmented = pipeline(vol)
```

---

## 5. CT pipeline with window/level and beam hardening

```python
from medaugment import Compose
from medaugmentx.transforms import (
    RandomAffine, ElasticDeform, RandomFlip,
    WindowLevel, GaussianNoise, GammaCorrection,
    BeamHardening,
)

ct_pipeline = Compose([
    RandomFlip(axes=("x",), p_per_axis=0.5, p=0.5),
    RandomAffine(rotation=8, scale=(0.9, 1.1), translation=(-0.03, 0.03), p=0.7),
    ElasticDeform(alpha=20.0, sigma=4.0, p=0.4),
    WindowLevel(center_shift_frac=0.05, width_scale=(0.85, 1.15), p=0.6),
    GaussianNoise(std=(5.0, 20.0), p=0.4),
    GammaCorrection(gamma=(0.9, 1.1), p=0.4),
    BeamHardening(alpha=(0.02, 0.07), p=0.3),         # cupping artifact
], seed=1)
```

---

## 6. Tomosynthesis (DBT) pipeline

```python
from medaugmentx.presets import dbt_pipeline

pipeline = dbt_pipeline(seed=0)
augmented = pipeline(dbt_vol)
```

Or assembled manually:

```python
from medaugment import Compose
from medaugmentx.transforms import (
    RandomFlip, RandomAffine, AnisotropicElastic,
    SlabShift, LimitedAngleBlur, SliceDropout,
    BiasField, GammaCorrection,
)

pipeline = Compose([
    RandomFlip(axes=("x",), p_per_axis=0.5),
    RandomAffine(rotation=5, scale=(0.95, 1.05), axes_enabled=("x", "y"), p=0.7),
    AnisotropicElastic(alpha=(80, 80, 6), sigma=(8, 8, 2), p=0.5),
    SlabShift(max_shift=2, p=0.5),
    LimitedAngleBlur(arc_degrees=(15.0, 25.0), base_sigma=1.0, p=0.6),
    SliceDropout(num_slices=(1, 2), p=0.3),
    BiasField(alpha=0.2, coarse_shape=3, p=0.5),
    GammaCorrection(gamma=(0.85, 1.15), p=0.5),
], seed=0)
```

---

## 7. Load a DICOM series or NIfTI file

```python
from medaugmentx.io import load_dicom_series, load_nifti, save_nifti

vol_ct  = load_dicom_series("/data/studies/12345/CT_chest/")  # MedVolume
vol_mri = load_nifti("brain_t1.nii.gz")                        # MedVolume

augmented = ct_pipeline(vol_ct)
save_nifti(augmented, "ct_augmented.nii.gz")
```

Both loaders populate `spacing` and `metadata` (modality, vendor, DICOM tags)
automatically.

> Requires the `io` extra: `pip install "medaugmentx[io]"`

---

## 8. Serialise a pipeline to JSON and reload it

```python
from medaugmentx.presets import mri_pipeline
from medaugmentx.serialization import to_json, from_json

pipeline = mri_pipeline(seed=42)

# Serialise
json_str = to_json(pipeline)
with open("mri_pipeline.json", "w") as f:
    f.write(json_str)

# Reload
pipeline2 = from_json(open("mri_pipeline.json").read())
out = pipeline2(vol)
```

Optional YAML round-trip (requires `pip install pyyaml`):

```python
from medaugmentx.serialization import to_yaml, from_yaml

yaml_str = to_yaml(pipeline)
pipeline3 = from_yaml(yaml_str)
```

---

## 9. Use with PyTorch `Dataset` / `DataLoader`

PyTorch is **not** a MedAugment dependency. Wire it up in your own dataset:

```python
from torch.utils.data import Dataset
from medaugmentx.io import load_nifti
from medaugmentx.presets import mri_pipeline

class MRIVolumes(Dataset):
    def __init__(self, paths):
        self.paths = paths
        # No seed — each call to __getitem__ draws fresh augmentations.
        # The pipeline's RNG advances independently on every call, so samples
        # within a batch and across epochs are all different.
        self.pipeline = mri_pipeline(seed=None)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        vol = load_nifti(self.paths[idx])
        vol = self.pipeline(vol)
        # PyTorch wants channel-first — add a channel axis
        return vol.image[None], (vol.mask if vol.mask is not None else 0)
```

For **reproducible epochs** (same augmentations every time epoch *N* runs),
create a fresh seeded pipeline at the start of each epoch rather than reusing
the same instance:

```python
# In your training loop:
for epoch in range(num_epochs):
    dataset.pipeline = mri_pipeline(seed=epoch)
    # ... train ...
```

A single `mri_pipeline(seed=42)` instance shared across all calls is
**not** the right pattern for per-epoch reproducibility — its internal RNG
advances with every sample, so the augmentations seen in epoch 2 depend on
how many samples were drawn in epoch 1.

---

## 10. Author your own transform

```python
from typing import Any
from medaugmentx.core import Transform, MedVolume

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
```

Register it for serialisation if needed:

```python
from medaugmentx.serialization import REGISTRY
REGISTRY["IntensityShift"] = IntensityShift
```

Drop it into a `Compose` like any built-in transform. Always sample from
`self.rng`, never from `np.random` — see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 11. Inspect and introspect a pipeline

```python
print(pipeline)
# Compose(transforms=[RandomAffine(...), BiasField(...), ...], p=1.0)

# Dict form — suitable for logging or passing to from_dict()
import json
print(json.dumps(pipeline.to_dict(), indent=2, default=str))
```

---

## 12. Reproducibility check

```python
import numpy as np
from medaugment import Compose
from medaugmentx.transforms import GaussianNoise, GammaCorrection

# Same instance — RNG advances each call, so output differs
a = pipeline(vol)
b = pipeline(vol)
assert not np.array_equal(a.image, b.image)   # different outputs

# Same seed — always identical
c = Compose([GaussianNoise(std=0.05), GammaCorrection()], seed=42)(vol)
d = Compose([GaussianNoise(std=0.05), GammaCorrection()], seed=42)(vol)
assert np.array_equal(c.image, d.image)        # always passes
```
