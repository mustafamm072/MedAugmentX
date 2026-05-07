# MedAugment

**Clinically-aware medical image augmentation for AI training.**

[![Python](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Phase 1](https://img.shields.io/badge/status-phase%201%20MVP-orange.svg)](docs/MILESTONES.md)

MedAugment is a purpose-built Python library for data augmentation of medical
images. Unlike general-purpose libraries (`torchvision`, `albumentations`), it
treats the unique properties of medical data — anisotropic 3D volumes,
modality-specific artifacts, mask consistency, and clinical I/O — as
first-class concerns.

> **Phase 1 (MVP)** — this release ships the core data model, pipeline
> primitives, the most-used spatial / intensity transforms, the first
> tomosynthesis (DBT) transforms, and unified DICOM / NIfTI I/O. The full
> roadmap lives in [docs/MILESTONES.md](docs/MILESTONES.md).

---

## Why MedAugment

| Problem with general-purpose augmentation | What MedAugment does |
| --- | --- |
| Treats every image as 2D RGB | Native 3D volumes with anisotropic-aware ops |
| No MRI-specific noise models | Physics-based (Rician, etc.) noise per modality |
| Masks drift on complex transforms | Deterministic seeding; image and mask share the same realisation |
| No tomosynthesis support | Dedicated DBT module with slab-aware augmentations |
| DICOM/NIfTI I/O scattered across libraries | One loader, one `MedVolume`, vendor metadata preserved |

## Install

```bash
# Core (NumPy + SciPy only)
pip install medaugment

# With DICOM/NIfTI I/O
pip install "medaugment[io]"
```

The core library has **zero deep-learning dependencies**. PyTorch, MONAI, and
TorchIO interop arrive in Phase 3.

## Quick start

```python
import numpy as np
from medaugment import MedVolume, Compose, OneOf
from medaugment.transforms import (
    RandomAffine, RandomFlip, ElasticDeform,
    RicianNoise, GaussianNoise, GammaCorrection,
)

# Wrap your image (and optional mask) once.
vol = MedVolume(
    image=np.random.rand(80, 256, 256).astype(np.float32),
    mask=np.zeros((80, 256, 256), dtype=np.uint8),
    spacing=(1.0, 0.7, 0.7),               # mm per axis
    metadata={"modality": "MR"},
)

augment = Compose([
    RandomAffine(rotation=15, scale=(0.9, 1.1), p=0.7),
    ElasticDeform(alpha=(120, 120, 10), sigma=(10, 10, 3), p=0.5),
    OneOf([
        RicianNoise(std=0.02),
        GaussianNoise(std=0.015),
    ], p=0.6),
    GammaCorrection(gamma=(0.8, 1.2), p=0.5),
], seed=42)

out = augment(vol)            # MedVolume in, MedVolume out
print(out.image.shape, out.mask.shape, out.spacing)
```

Mask values are preserved exactly — spatial transforms use nearest-neighbour
interpolation for the mask and the same random sample as the image.

### Tomosynthesis (DBT) preset — Phase 1 transforms only

```python
from medaugment import Compose
from medaugment.transforms import (
    RandomFlip, AnisotropicElastic,
    SlabShift, LimitedAngleBlur, SliceDropout,
    GammaCorrection,
)

dbt_pipeline = Compose([
    RandomFlip(axes=("x",), p=0.5),
    SlabShift(max_shift=2, p=0.5),
    AnisotropicElastic(alpha=(100, 100, 8), sigma=(8, 8, 2), p=0.4),
    LimitedAngleBlur(arc_degrees=20, p=0.3),
    SliceDropout(num_slices=(1, 2), p=0.2),
    GammaCorrection(gamma=(0.85, 1.15), p=0.4),
], seed=0)
```

Full preset pipelines (`TOMOSYNTHESIS_STANDARD`, `MRI_STANDARD`, …) and YAML
serialisation arrive in Phase 2.

### Loading a real volume

```python
from medaugment.io import load_dicom_series, load_nifti

vol_ct  = load_dicom_series("/path/to/study/")        # MedVolume
vol_mri = load_nifti("brain_t1.nii.gz")                # MedVolume
```

Both helpers populate `spacing` and `metadata` (modality, vendor, original
DICOM tags) automatically.

## What's in Phase 1

| Layer | Components |
| --- | --- |
| Core | `MedVolume`, `Transform` ABC, `Compose`, `OneOf`, `SomeOf`, RNG helpers |
| Spatial | `RandomAffine`, `RandomFlip`, `AnatomicCrop`, `ElasticDeform` |
| Intensity | `RicianNoise`, `GaussianNoise`, `GammaCorrection` |
| Tomosynthesis | `SlabShift`, `LimitedAngleBlur`, `SliceDropout`, `AnisotropicElastic` |
| I/O | DICOM series loader, NIfTI reader/writer |
| Quality | pytest suite, type hints throughout, GitHub Actions CI |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the layered design and
[docs/API_EXAMPLES.md](docs/API_EXAMPLES.md) for end-to-end recipes.

## Reproducibility

Every transform is seedable. `Compose(..., seed=42)` produces bit-identical
output across runs and across machines (within a single NumPy version):

```python
out_a = Compose([...], seed=42)(vol)
out_b = Compose([...], seed=42)(vol)
assert np.array_equal(out_a.image, out_b.image)
```

## Project status & roadmap

| Phase | Status |
| --- | --- |
| **1 — Core MVP** | ✅ This release |
| 2 — Modality artifacts (MRI ghosting, CT beam hardening, X-Ray scatter, full DBT, YAML pipelines) | Planned |
| 3 — GPU backend, framework interop, benchmark suite, Sphinx docs, v1.0 | Planned |

Detailed deliverables: [docs/MILESTONES.md](docs/MILESTONES.md).

## Contributing

We welcome PRs — especially new transforms, vendor DICOM coverage, and bug
reports with sample data. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, testing conventions, and the transform-authoring template.

## License

MIT — see [LICENSE](LICENSE).
