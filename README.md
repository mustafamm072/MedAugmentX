# MedAugmentX

**Clinically-aware medical image augmentation for AI training.**

[![Python](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Phase 2](https://img.shields.io/badge/status-phase%202-green.svg)](docs/MILESTONES.md)

MedAugmentX is a purpose-built Python library for data augmentation of medical
images. Unlike general-purpose libraries (`torchvision`, `albumentations`), it
treats the unique properties of medical data — anisotropic 3D volumes,
modality-specific artifacts, mask consistency, and clinical I/O — as
first-class concerns.

**Zero deep-learning dependencies.** The core library requires only NumPy and
SciPy. PyTorch, MONAI, and TorchIO interop are planned for Phase 3.

---

## Why MedAugmentX

| Problem with general-purpose augmentation | What MedAugmentX does |
| --- | --- |
| Treats every image as 2D RGB | Native 3D volumes with anisotropic-aware ops |
| No MRI-specific noise or artifact models | Physics-based Rician noise, bias field, k-space corruption, ghosting |
| CT/DXR augmentation is generic brightness shift | Window/level variation, beam hardening, resolution simulation |
| Masks drift on complex transforms | Deterministic seeding; image and mask share the same random draw |
| No tomosynthesis support | Dedicated DBT module with slab-aware augmentations |
| DICOM/NIfTI I/O scattered across libraries | One loader, one `MedVolume`, vendor metadata preserved |
| Pipelines can't be saved or reloaded | Full JSON/YAML serialisation with round-trip reconstruction |

---

## Install

```bash
# Core (NumPy + SciPy only)
pip install medaugmentx

# With DICOM/NIfTI I/O
pip install "medaugmentx[io]"

# With YAML serialisation support
pip install "medaugmentx[yaml]"
```

Verify the installation:

```python
import medaugmentx
print(medaugmentx.__version__)   # 0.2.0
```

---

## Quick start

### One-line preset

The fastest way to get started is a pre-built modality preset:

```python
from medaugmentx.presets import mri_pipeline, ct_pipeline, dxr_pipeline, dbt_pipeline

pipeline = mri_pipeline(seed=42)
augmented = pipeline(vol)          # MedVolume in, MedVolume out
```

Four presets ship out of the box — see [Preset pipelines](#preset-pipelines) below.

### Build your own pipeline

```python
import numpy as np
from medaugmentx import MedVolume, Compose, OneOf
from medaugmentx.transforms import (
    RandomAffine, ElasticDeform,
    BiasField, RicianNoise, GaussianNoise, GammaCorrection,
)

vol = MedVolume(
    image=np.random.rand(80, 256, 256).astype(np.float32),
    mask=np.zeros((80, 256, 256), dtype=np.uint8),
    spacing=(1.0, 0.7, 0.7),          # mm per axis (z, y, x)
    metadata={"modality": "MR"},
)

augment = Compose([
    RandomAffine(rotation=15, scale=(0.9, 1.1), p=0.7),
    ElasticDeform(alpha=(120, 120, 10), sigma=(10, 10, 3), p=0.5),
    BiasField(alpha=0.3, p=0.7),
    OneOf([
        RicianNoise(std=(0.005, 0.02)),   # MRI physical noise model
        GaussianNoise(std=0.015),          # simpler additive noise
    ], p=0.5),
    GammaCorrection(gamma=(0.8, 1.2), p=0.5),
], seed=42)

out = augment(vol)           # mask kept in sync automatically
print(out.image.shape, out.mask.shape)
```

Mask values are preserved exactly — spatial transforms use nearest-neighbour
interpolation for the mask and the same random draw as the image.

---

## Preset pipelines

Ready-to-use `Compose` pipelines for common modalities. All presets are
seeded, deterministic, and serialisable.

```python
from medaugmentx.presets import mri_pipeline, ct_pipeline, dxr_pipeline, dbt_pipeline

mri  = mri_pipeline(seed=42)   # bias field, Rician noise, ghosting/k-space
ct   = ct_pipeline(seed=42)    # window/level, Gaussian noise, beam hardening
dxr  = dxr_pipeline(seed=42)  # blur, brightness/contrast, low-resolution sim
dbt  = dbt_pipeline(seed=42)  # slab shift, limited-angle blur, anisotropic elastic
```

Each preset is a starting point — tune the parameters or swap transforms for
your dataset.

---

## Serialisation

Pipelines serialise to JSON (built-in) or YAML (optional PyYAML):

```python
from medaugmentx.serialization import to_json, from_json

pipeline = mri_pipeline(seed=42)

# Save
json_str = to_json(pipeline)
with open("pipeline.json", "w") as f:
    f.write(json_str)

# Reload — exact same structure, ready to run
pipeline2 = from_json(open("pipeline.json").read())
out = pipeline2(vol)
```

Custom transforms can be added to the registry so they serialise too:

```python
from medaugmentx.serialization import REGISTRY
REGISTRY["MyTransform"] = MyTransform
```

---

## Loading real volumes

```python
from medaugmentx.io import load_dicom_series, load_nifti, save_nifti

vol_ct  = load_dicom_series("/path/to/study/CT_chest/")  # MedVolume
vol_mri = load_nifti("brain_t1.nii.gz")                   # MedVolume

augmented = ct_pipeline(seed=0)(vol_ct)
save_nifti(augmented, "ct_augmented.nii.gz")
```

Both loaders populate `spacing` and `metadata` (modality, vendor, DICOM tags)
automatically.

> Requires the `io` extra: `pip install "medaugmentx[io]"`

---

## What's available

### Core

| Component | Description |
| --- | --- |
| `MedVolume` | Container: image + optional mask + spacing + metadata |
| `Transform` | Abstract base — probability gating, seedable RNG, `to_dict()` |
| `Compose` | Sequential pipeline with deterministic child seeding |
| `OneOf` / `SomeOf` | Random selection from a set of transforms |

### Spatial transforms

| Transform | What it does |
| --- | --- |
| `RandomAffine` | Rotation + scaling + translation, 2D/3D, axis-aware |
| `RandomFlip` | Per-axis flip with independent probability |
| `AnatomicCrop` | Foreground-biased random crop |
| `ElasticDeform` | Anisotropic B-spline elastic deformation |

### Intensity transforms

| Transform | What it does |
| --- | --- |
| `GaussianNoise` | Additive zero-mean Gaussian noise |
| `RicianNoise` | MRI magnitude noise (physically correct for low-SNR regions) |
| `GammaCorrection` | Power-law contrast, normalisation-aware |
| `BiasField` | Smooth multiplicative field (MRI coil inhomogeneity) |
| `WindowLevel` | Random CT window/level shift (protocol variation) |
| `BrightnessContrast` | Additive brightness + multiplicative contrast |
| `GaussianBlur` | Isotropic Gaussian blur |
| `SimulateLowResolution` | Downsample + upsample (cross-site resolution variation) |

### Modality-specific — MRI

| Transform | What it does |
| --- | --- |
| `GhostingArtifact` | Phase-encoding ghosting (shifted attenuated replica) |
| `KSpaceDropout` | Random k-space line zeroing with Gibbs ringing reconstruction |

### Modality-specific — CT

| Transform | What it does |
| --- | --- |
| `BeamHardening` | Radially-symmetric cupping artifact |

### Modality-specific — Tomosynthesis (DBT)

| Transform | What it does |
| --- | --- |
| `SlabShift` | Z-axis reconstruction-centre variation |
| `LimitedAngleBlur` | Arc-angle-dependent Z-only blur |
| `SliceDropout` | Random slice zeroing (robustness) |
| `AnisotropicElastic` | DBT-default elastic deformation |

### I/O

| Helper | Description |
| --- | --- |
| `load_dicom_series(path)` | 3D volume from a DICOM series directory |
| `load_nifti(path)` | MedVolume from `.nii` / `.nii.gz` |
| `save_nifti(vol, path)` | Write MedVolume to NIfTI |

---

## Reproducibility

Every transform is seedable. `Compose(..., seed=42)` produces bit-identical
output across runs and across machines (within a NumPy version):

```python
a = Compose([...], seed=42)(vol)
b = Compose([...], seed=42)(vol)
assert np.array_equal(a.image, b.image)  # always passes
```

---

## Project status & roadmap

| Phase | Status |
| --- | --- |
| **1 — Core MVP** | ✅ Core data model, spatial/intensity transforms, DBT, DICOM/NIfTI I/O |
| **2 — Modality artifacts & serialisation** | ✅ MRI (bias field, ghosting, k-space), CT (beam hardening), presets, JSON/YAML |
| **3 — GPU backend, framework interop, v1.0** | Planned |

Detailed deliverables: [docs/MILESTONES.md](docs/MILESTONES.md).

---

## Contributing

We welcome PRs — new transforms, vendor DICOM coverage, and bug reports with
sample data are especially valuable. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the development setup, testing conventions, and the transform-authoring
template.

## License

MIT — see [LICENSE](LICENSE).
