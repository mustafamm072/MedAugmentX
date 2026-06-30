# MedAugmentX

**Clinically-aware medical image augmentation for AI training.**

[![DOI](https://zenodo.org/badge/1231536084.svg)](https://doi.org/10.5281/zenodo.20191189)
[![Python](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Phase 3](https://img.shields.io/badge/status-phase%203-blue.svg)](docs/MILESTONES.md)

MedAugmentX is a purpose-built Python library for data augmentation of medical
images. Unlike general-purpose libraries (`torchvision`, `albumentations`), it
treats the unique properties of medical data — anisotropic 3D volumes,
modality-specific artifacts, mask consistency, and clinical I/O — as
first-class concerns.

**Zero deep-learning dependencies in the core install.** The library requires
only NumPy and SciPy by default. PyTorch, MONAI, and TorchIO remain optional
extras, with lightweight dataset adapters available in `medaugmentx.interop`.

MedAugmentX is intended for research and training-data augmentation workflows.
It is not a diagnostic device, clinical decision-support system, or substitute
for local clinical, regulatory, security, and dataset validation.

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

## Commercial Readiness

MedAugmentX is designed to be easy to evaluate inside serious medical AI
teams:

- **Lightweight core:** only NumPy and SciPy are required by default.
- **Optional integrations:** PyTorch, MONAI, TorchIO, DICOM, NIfTI, and YAML
  support stay behind extras.
- **Reproducible pipelines:** every transform is seedable and serialisable.
- **Mask-safe spatial ops:** image and mask geometry share the same sampled
  transform parameters.
- **Typed package:** ships `py.typed` for downstream type checkers.
- **Clear adoption guidance:** see [Commercial adoption](docs/COMMERCIAL_ADOPTION.md),
  [Documentation](docs/README.md), and [Security](SECURITY.md).

Teams using MedAugmentX in regulated or customer-facing products should
validate augmentation policies on their own data, version-control pipeline
JSON/YAML, and document intended use before deployment.

---

## Install

```bash
# Core (NumPy + SciPy only)
pip install medaugmentx

# With DICOM/NIfTI I/O
pip install "medaugmentx[io]"

# With YAML serialisation support
pip install "medaugmentx[yaml]"

# Optional framework extras
pip install "medaugmentx[torch]"      # PyTorch / torchvision tensors
pip install "medaugmentx[monai]"      # MONAI projects
pip install "medaugmentx[torchio]"    # TorchIO subjects
pip install "medaugmentx[frameworks]" # PyTorch + MONAI + TorchIO
```

Verify the installation:

```python
import medaugmentx
print(medaugmentx.__version__)   # 0.7.0
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

Custom transforms can be registered so they serialise too. The
`@register_transform` decorator validates the class and refuses to silently
shadow a built-in name:

```python
from medaugmentx.core import Transform
from medaugmentx.serialization import register_transform

@register_transform
class MyTransform(Transform):
    ...
```

Direct assignment (`REGISTRY["MyTransform"] = MyTransform`) still works.

---

## Pipeline inspection

Use `pipeline_summary` to log or review a compact tree of any transform or
nested pipeline before saving it with your experiment artifacts:

```python
from medaugmentx import pipeline_summary
from medaugmentx.presets import mri_pipeline

print(pipeline_summary(mri_pipeline(seed=42)))
```

Example output:

```text
Compose(p=1.0, seed=42)
  0 RandomFlip(...)
  1 RandomAffine(...)
  2 OneOf(...)
    2.0 GhostingArtifact(...)
    2.1 KSpaceDropout(...)
```

For programmatic inspection, `iter_pipeline(pipeline)` yields `PipelineStep`
objects with `path`, `name`, `params`, and `depth`.

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

## Framework interop

Use `TorchTransform` when your dataset returns tensors, NumPy arrays, tuples,
or dict samples. The adapter keeps PyTorch optional: tensors are supported at
runtime, but importing MedAugmentX never imports torch.

```python
from medaugmentx.interop import TorchTransform
from medaugmentx.presets import mri_pipeline

augment = TorchTransform(
    mri_pipeline(seed=None),
    image_key="image",
    mask_key="mask",
    channel_dim=0,        # handles (1, D, H, W) tensors
)

sample = {"image": image_tensor, "mask": mask_tensor, "spacing": (1.0, 0.7, 0.7)}
sample = augment(sample)
```

For MONAI-style dicts that use `label` instead of `mask`:

```python
from medaugmentx.interop import MonaiMapTransform

augment = MonaiMapTransform(mri_pipeline(seed=None), image_key="image", label_key="label")
```

For TorchIO subjects:

```python
from medaugmentx.interop import TorchIOTransform

augment = TorchIOTransform(mri_pipeline(seed=None), image_key="t1", label_key="seg")
subject = augment(subject)
```

See [API examples](docs/API_EXAMPLES.md) and the [API reference](docs/API_REFERENCE.md).

---

## What's available

### Core

| Component | Description |
| --- | --- |
| `MedVolume` | Container: image + optional mask + spacing + metadata |
| `Transform` | Abstract base — probability gating, seedable RNG, `to_dict()` |
| `Compose` | Sequential pipeline with deterministic child seeding |
| `OneOf` / `SomeOf` | Random selection from a set of transforms |
| `pipeline_summary` / `iter_pipeline` | Human-readable and programmatic pipeline inspection |

### Spatial transforms

| Transform | What it does |
| --- | --- |
| `RandomAffine` | Rotation + scaling + translation, 2D/3D, axis-aware |
| `RandomFlip` | Per-axis flip with independent probability |
| `AnatomicCrop` | Foreground-biased random crop |
| `ElasticDeform` | Anisotropic B-spline elastic deformation |
| `CoarseDropout` | Cutout-style random box occlusion (2D/3D) |
| `Resize` | Resample to a fixed shape (rescales spacing) |
| `Pad` | Centre-pad up to a target shape |
| `CenterCrop` | Centre-crop to a target shape |

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
| `MedianBlur` | Edge-preserving median filter (salt-and-pepper / speckle) |
| `SimulateLowResolution` | Downsample + upsample (cross-site resolution variation) |
| `Sharpen` | Unsharp-mask edge enhancement |
| `CLAHEContrast` | Contrast Limited Adaptive Histogram Equalization |
| `HistogramMatch` | Match intensity histogram to a reference distribution |

### Modality-specific — MRI

| Transform | What it does |
| --- | --- |
| `GhostingArtifact` | Phase-encoding ghosting (shifted attenuated replica) |
| `KSpaceDropout` | Random k-space line zeroing with Gibbs ringing reconstruction |
| `MRIMotion` | In-plane rigid motion blur/ghosting (averaged motion states) |

### Modality-specific — CT

| Transform | What it does |
| --- | --- |
| `BeamHardening` | Radially-symmetric cupping artifact |
| `MetalStreak` | Radiating bright/dark streaks from dense implants |

### Modality-specific — X-ray (DXR)

| Transform | What it does |
| --- | --- |
| `ScatterSimulation` | Low-frequency scatter (veiling glare), lowers contrast |
| `GridArtifact` | Stationary anti-scatter grid line pattern |

### Modality-specific — Tomosynthesis (DBT)

| Transform | What it does |
| --- | --- |
| `SlabShift` | Z-axis reconstruction-centre variation |
| `LimitedAngleBlur` | Arc-angle-dependent Z-only blur |
| `SliceDropout` | Random slice zeroing (robustness) |
| `AnisotropicElastic` | DBT-default elastic deformation |
| `CompressionVariation` | Breast-paddle compression variation (mask-consistent) |
| `ReconStreak` | Limited-angle out-of-plane parallax replicas |

### I/O

| Helper | Description |
| --- | --- |
| `load_dicom_series(path)` | 3D volume from a DICOM series directory |
| `load_nifti(path)` | MedVolume from `.nii` / `.nii.gz` |
| `save_nifti(vol, path)` | Write MedVolume to NIfTI |

### Interop

| Adapter | Description |
| --- | --- |
| `SampleTransform` | Generic adapter for arrays, tensors, tuples, dicts, and `MedVolume` |
| `TorchTransform` | PyTorch / torchvision-friendly alias for dataset samples |
| `MonaiMapTransform` | MONAI-style dict adapter with `image` / `label` defaults |
| `TorchIOTransform` | TorchIO `Subject` adapter with optional image/label key inference |

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
| **3 — Framework interop, GPU backend, v1.0** | In progress: `0.7.0` adds pipeline inspection for experiment logs and policy review; `0.6.0` added 14 transforms and a new X-ray modality module; `0.5.0` added custom-transform registration; `0.4.0` shipped TorchIO interop |

Detailed deliverables: [docs/MILESTONES.md](docs/MILESTONES.md).
Developer API: [docs/API_REFERENCE.md](docs/API_REFERENCE.md).

---

## Contributing

We welcome PRs — new transforms, vendor DICOM coverage, and bug reports with
sample data are especially valuable. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the development setup, testing conventions, and the transform-authoring
template.

## License

MIT — see [LICENSE](LICENSE).
