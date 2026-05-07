# API Examples

Working code for the most common Phase 1 use-cases. Every snippet is
self-contained — copy, paste, run.

## 1. Build a `MedVolume` from a NumPy array

```python
import numpy as np
from medaugment import MedVolume

vol = MedVolume(
    image=np.random.rand(80, 256, 256).astype(np.float32),  # (D, H, W)
    mask=np.zeros((80, 256, 256), dtype=np.uint8),
    spacing=(1.0, 0.7, 0.7),                                 # mm per axis
    metadata={"modality": "MR", "patient_id": "anon-001"},
)
print(vol)
# MedVolume(image=shape=(80, 256, 256), dtype=float32, mask=shape=(80, 256, 256), ...)
```

## 2. Apply a single transform

```python
from medaugment.transforms import RandomFlip

augmented = RandomFlip(axes=("x",), p_per_axis=1.0, seed=0)(vol)
```

## 3. Compose a pipeline (deterministic seeding)

```python
from medaugment import Compose, OneOf
from medaugment.transforms import (
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
output every time, on every machine.

## 4. Tomosynthesis-tuned pipeline (Phase 1 transforms)

```python
from medaugment import Compose
from medaugment.transforms import (
    RandomFlip, AnisotropicElastic,
    SlabShift, LimitedAngleBlur, SliceDropout,
    GammaCorrection,
)

dbt_pipeline = Compose([
    RandomFlip(axes=("x",), p_per_axis=0.5),
    SlabShift(max_shift=2, p=0.5),
    AnisotropicElastic(alpha=(100, 100, 8), sigma=(8, 8, 2), p=0.4),
    LimitedAngleBlur(arc_degrees=(15.0, 25.0), base_sigma=1.0, p=0.3),
    SliceDropout(num_slices=(1, 2), p=0.2),
    GammaCorrection(gamma=(0.85, 1.15), p=0.4),
], seed=0)
```

## 5. Load a DICOM series

```python
from medaugment.io import load_dicom_series

vol = load_dicom_series("/data/studies/12345/CT_chest/")
print(vol.spacing)           # (z_mm, y_mm, x_mm)
print(vol.metadata["vendor"])
print(vol.metadata["modality"])
```

The loader sorts slices by image position projected onto the slice normal,
applies `RescaleSlope * pixels + RescaleIntercept`, and rejects mixed
SeriesInstanceUIDs in one folder.

> Requires the `dicom` extra: `pip install "medaugment[dicom]"`.

## 6. Load and save NIfTI

```python
from medaugment.io import load_nifti, save_nifti

vol = load_nifti("brain_t1.nii.gz")           # MedVolume
augmented = pipeline(vol)
save_nifti(augmented, "brain_t1_augmented.nii.gz")
```

> Requires the `nifti` extra: `pip install "medaugment[nifti]"`.

## 7. Use with PyTorch `Dataset` / `DataLoader`

PyTorch is **not** a MedAugment dependency. Wire it up in your own
dataset class:

```python
from torch.utils.data import Dataset
from medaugment.io import load_nifti

class MRIVolumes(Dataset):
    def __init__(self, paths, pipeline):
        self.paths = paths
        self.pipeline = pipeline

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        vol = load_nifti(self.paths[idx])
        vol = self.pipeline(vol)
        # PyTorch wants channel-first, so add a channel axis.
        return vol.image[None], (vol.mask if vol.mask is not None else 0)
```

Pass a different `seed` per-epoch to `Compose` if you want fresh
augmentations between epochs while keeping each epoch reproducible.

## 8. Author your own transform

```python
from medaugment.core import Transform, MedVolume

class IntensityShift(Transform):
    """Add a uniform random offset to all voxels."""

    def __init__(self, max_shift=0.05, p=1.0, seed=None):
        super().__init__(p=p, seed=seed)
        self.max_shift = float(max_shift)

    def apply(self, volume: MedVolume) -> MedVolume:
        delta = self.rng.uniform(-self.max_shift, self.max_shift)
        return volume.replace(image=volume.image + delta)
```

Drop it into a `Compose` like any built-in transform. Always sample from
`self.rng`, never from `np.random` — see [docs/ARCHITECTURE.md](ARCHITECTURE.md).

## 9. Inspect a pipeline

```python
print(pipeline)
# Compose(transforms=[RandomAffine(...), ElasticDeform(...), OneOf(...), GammaCorrection(...)], p=1.0)

print([t.to_dict() for t in pipeline])
# [{'name': 'RandomAffine', 'params': {...}}, ...]
```

`to_dict()` is best-effort introspection; full YAML round-trip serialisation
arrives in Phase 2.

## 10. Reproducibility check

```python
import numpy as np

a = pipeline(vol)
b = pipeline(vol)  # second call — same instance, different RNG state, different output
c = Compose([...], seed=42)(vol)
d = Compose([...], seed=42)(vol)
assert np.array_equal(c.image, d.image)  # passes
assert not np.array_equal(a.image, b.image)  # passes
```
