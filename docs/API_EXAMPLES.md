# API Examples

Working patterns for common use-cases. Snippets that use real DICOM, NIfTI,
TorchIO, or training data paths are intentionally minimal; replace paths and
dataset objects with your own local data.

---

## 1. Build a `MedVolume` from a NumPy array

```python
import numpy as np
from medaugmentx import MedVolume

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
from medaugmentx import Compose, OneOf
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

## 4. MRI pipeline with modality transforms

```python
from medaugmentx import Compose, OneOf
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
from medaugmentx import Compose
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
from medaugmentx import Compose
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
from medaugmentx.presets import ct_pipeline

vol_ct  = load_dicom_series("/data/studies/12345/CT_chest/")  # MedVolume
vol_mri = load_nifti("brain_t1.nii.gz")                        # MedVolume

augmented = ct_pipeline(seed=0)(vol_ct)
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

Optional YAML round-trip (requires `pip install "medaugmentx[yaml]"`):

```python
from medaugmentx.serialization import to_yaml, from_yaml

yaml_str = to_yaml(pipeline)
pipeline3 = from_yaml(yaml_str)
```

---

## 9. Use with PyTorch `Dataset` / `DataLoader`

PyTorch is **not** a MedAugmentX core dependency. Wrap any MedAugmentX
pipeline with `TorchTransform` when your dataset returns tensors, NumPy
arrays, `(image, mask)` tuples, or dict samples:

```python
from torch.utils.data import Dataset
from medaugmentx.interop import TorchTransform
from medaugmentx.io import load_nifti
from medaugmentx.presets import mri_pipeline

class MRIVolumes(Dataset):
    def __init__(self, paths):
        self.paths = paths
        pipeline = mri_pipeline(seed=None)
        self.augment = TorchTransform(
            pipeline,
            image_key="image",
            mask_key="mask",
            channel_dim=0,      # strips/restores a singleton channel axis
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        vol = load_nifti(self.paths[idx])
        sample = {
            "image": vol.image[None],  # (1, D, H, W)
            "mask": None if vol.mask is None else vol.mask[None],
            "spacing": vol.spacing,
            "metadata": vol.metadata,
        }
        return self.augment(sample)
```

For **reproducible epochs** (same augmentations every time epoch *N* runs),
create a fresh seeded pipeline at the start of each epoch rather than reusing
the same instance:

```python
# In your training loop:
for epoch in range(num_epochs):
    dataset.augment = TorchTransform(mri_pipeline(seed=epoch), channel_dim=0)
    # ... train ...
```

A single `mri_pipeline(seed=42)` instance shared across all calls is
**not** the right pattern for per-epoch reproducibility — its internal RNG
advances with every sample, so the augmentations seen in epoch 2 depend on
how many samples were drawn in epoch 1.

---

## 10. Use with MONAI-style dictionary samples

`MonaiMapTransform` defaults to `image` and `label` keys and can be used in
MONAI-style transform pipelines without making MONAI a required dependency:

```python
from medaugmentx.interop import MonaiMapTransform
from medaugmentx.presets import ct_pipeline

augment = MonaiMapTransform(
    ct_pipeline(seed=None),
    image_key="image",
    label_key="label",
    channel_dim=0,
)

sample = {"image": image_tensor, "label": label_tensor, "spacing": (1.0, 0.7, 0.7)}
sample = augment(sample)
```

Masks/labels preserve dtype by default. Image dtype follows the transform
output unless you pass `preserve_image_dtype=True`.

---

## 11. Use with TorchIO subjects

TorchIO is optional. `TorchIOTransform` uses duck typing, so importing the
adapter does not import TorchIO until your own code creates TorchIO objects:

```python
import torchio as tio
from medaugmentx.interop import TorchIOTransform
from medaugmentx.presets import mri_pipeline

subject = tio.Subject(
    t1=tio.ScalarImage("t1.nii.gz"),
    seg=tio.LabelMap("seg.nii.gz"),
)

augment = TorchIOTransform(
    mri_pipeline(seed=None),
    image_key="t1",
    label_key="seg",
    channel_dim=0,      # TorchIO data is channel-first
)

subject = augment(subject)
```

If a subject has exactly one scalar-like image and one label-like image, the
keys can be inferred. Pass explicit keys for multi-contrast studies.

> Requires the `torchio` extra: `pip install "medaugmentx[torchio]"`

---

## 12. Author your own transform

```python
from typing import Any
from medaugmentx.core import Transform, MedVolume
from medaugmentx.serialization import register_transform

@register_transform   # registers it for JSON/YAML serialisation
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

The `@register_transform` decorator validates the class and refuses to silently
overwrite an existing registry entry. Direct assignment,
`REGISTRY["IntensityShift"] = IntensityShift`, also works.

Drop it into a `Compose` like any built-in transform. Always sample from
`self.rng`, never from `np.random` — see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 13. Inspect and introspect a pipeline

```python
from medaugmentx import iter_pipeline, pipeline_summary

print(pipeline_summary(pipeline))
# Compose(p=1.0, seed=42)
#   0 RandomAffine(...)
#   1 BiasField(...)

for step in iter_pipeline(pipeline):
    print(step.path, step.name, step.params)
```

---

## 14. Validate outputs and guard a pipeline

```python
from medaugmentx import Compose, Guard, VolumeValidator
from medaugmentx.transforms import RandomAffine, GammaCorrection

validator = VolumeValidator(
    intensity_bounds=(0.0, 1.0),   # warn if values leave the normalised range
    max_foreground_loss=0.5,       # error if a draw crops away >50% of the mask
    preserve_mask_labels=True,     # error if a labelled class disappears
)

# Audit a single volume (pass the original for comparative checks):
report = validator.validate(augmented, reference=original)
print(report.ok)          # False if any error-severity issue was found
print(report)             # readable list of errors + warnings

# Guard a pipeline: retry bad draws, fall back to the input if none pass.
safe = Guard(
    Compose([RandomAffine(), GammaCorrection()]),
    validator,
    on_fail="retry",      # "raise" | "warn" | "revert" | "retry"
    retries=3,
    seed=42,
)
out = safe(vol)           # guaranteed to pass the validator or return `vol` unchanged
```

---

## 15. Reproducibility check

```python
import numpy as np
from medaugmentx import Compose
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

---

## 16. Track keypoints and bounding boxes

```python
import numpy as np
from medaugmentx import Compose, MedVolume
from medaugmentx.transforms import RandomAffine, RandomFlip, Resize

# Coordinates are in array-index order: (y, x) for 2D, (z, y, x) for 3D.
vol = MedVolume(
    image=np.zeros((256, 256), dtype=np.float32),
    keypoints=np.array([[120.0, 80.0]]),             # (y, x)
    keypoint_labels=np.array(["landmark"]),
    bboxes=np.array([[105.0, 65.0, 135.0, 95.0]]),   # [y_min, x_min, y_max, x_max]
    bbox_labels=np.array(["lesion"]),
)

augment = Compose(
    [RandomFlip(axes=("x",)), RandomAffine(rotation=20), Resize((512, 512))],
    seed=7,
)
out = augment(vol)

out.keypoints        # moved with the anatomy; labels ride along unchanged
out.bboxes           # re-bounded to a valid axis-aligned box after rotation

# A crop can push targets off-frame; transforms keep faithful coordinates.
# Prune them explicitly when you need clean detection labels:
clean = out.remove_out_of_bounds_targets(min_visibility=0.25)
print(clean.num_keypoints, clean.num_bboxes)
```

Only spatial transforms move targets — intensity and artifact transforms
(and `CoarseDropout`) pass them through untouched.
