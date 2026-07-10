# Architecture

MedAugmentX is a layered library. Each layer has one responsibility, no
upward dependencies, and a flat public surface. This document explains the
layout and the rules each layer follows.

```
┌─────────────────────────────────────────────────────────────────┐
│ Interop  (medaugmentx/interop/)                                  │
│   SampleTransform · TorchTransform · MonaiMapTransform · TorchIO │
├─────────────────────────────────────────────────────────────────┤
│ Presets  (medaugmentx/presets.py)                                │
│   mri_pipeline · ct_pipeline · dxr_pipeline · dbt_pipeline     │
├─────────────────────────────────────────────────────────────────┤
│ Serialisation  (medaugmentx/serialization.py)                    │
│   REGISTRY · from_dict · to_json / from_json · to_yaml / from_yaml │
├─────────────────────────────────────────────────────────────────┤
│ Validation & inspection                                         │
│   VolumeValidator · Guard · pipeline_summary · iter_pipeline    │
├─────────────────────────────────────────────────────────────────┤
│ User pipeline  (Compose · OneOf · SomeOf)                       │
├─────────────────────────────────────────────────────────────────┤
│ Transforms                                                      │
│   spatial/   intensity/   modality/mri/   modality/ct/          │
│                           modality/xray/  modality/tomosynthesis/ │
├─────────────────────────────────────────────────────────────────┤
│ Core                                                            │
│   MedVolume · Transform ABC · seedable RNG · helpers            │
├─────────────────────────────────────────────────────────────────┤
│ I/O  (optional)                                                 │
│   DICOM series · NIfTI                                          │
└─────────────────────────────────────────────────────────────────┘
            numpy + scipy (always present)
            pydicom + nibabel (optional, behind extras)
            pyyaml (optional, behind [yaml] extra)
            torch + monai + torchio (optional, behind framework extras)
```

---

## 1. Core layer (`medaugmentx/core/`)

The foundational types. **Nothing else in the library may import upward into
user code, and core has no dependencies beyond NumPy.**

### `MedVolume` (`core/volume.py`)

```python
@dataclass
class MedVolume:
    image: np.ndarray                # (D, H, W) or (H, W), recommended float32
    mask: Optional[np.ndarray]       # same shape as image, integer labels
    spacing: Tuple[float, ...]       # mm per axis, one entry per ndim
    metadata: Dict[str, Any]         # modality, vendor, original tags...
```

Invariants enforced in `__post_init__`:

- image is 2D or 3D,
- mask, when present, has the same shape as image,
- spacing length matches image ndim (defaults to all-1.0 if omitted),
- metadata is a dict.

`replace()` returns a shallow copy with selected fields swapped. `copy()`
deep-copies the arrays.

### Axis convention

| Volume | Storage order | Axis 0 | Axis 1 | Axis 2 |
| --- | --- | --- | --- | --- |
| 3D | `(D, H, W)` | z (slice) | y (row) | x (col) |
| 2D | `(H, W)` | y (row) | x (col) | — |

`spacing` follows the same order. The DICOM and NIfTI loaders both transpose
to this convention so downstream code never has to think about it.

### `Transform` (`core/base.py`)

Abstract base class. Subclasses override `apply(volume) -> MedVolume`. The
base class handles:

- probability gating (`p` in `[0, 1]`),
- the `self.rng` generator (always used in transforms, never `np.random`),
- `to_dict()` — returns a dict suitable for reconstruction via
  `medaugmentx.serialization.from_dict()`.

### `Compose`, `OneOf`, `SomeOf` (`core/compose.py`)

- **`Compose`** runs children sequentially. The top-level seed is *spawned*
  into one independent generator per child via `derive_rng`, so a given seed
  produces bit-identical output every time. Adding or removing transforms
  does not change the seed assignments of the unchanged children.
- **`OneOf`** picks exactly one child, optionally weighted, and forces it to
  run regardless of the child's own `p`.
- **`SomeOf`** picks `n` (or a range) children without replacement and
  applies them in deterministic order.

All three containers override `to_dict()` to serialise their children
recursively, enabling lossless round-trip serialisation of entire pipelines.

### Seeding rules

1. Every transform owns a `np.random.Generator` (`self.rng`).
2. `Compose` derives child seeds from its own seed; passing a seed to the
   top level is sufficient for end-to-end reproducibility.
3. **Transforms must never call `np.random` directly.** Use `self.rng`.

---

## 2. Transforms layer (`medaugmentx/transforms/`)

Subdivided by *what* changes about the image:

| Folder | Changes | Contents |
| --- | --- | --- |
| `spatial/` | Geometry — pixels move | `RandomAffine`, `RandomFlip`, `AnatomicCrop`, `ElasticDeform`, `CoarseDropout`, `Resize`, `Pad`, `CenterCrop` |
| `intensity/` | Per-pixel value, geometry preserved | `GaussianNoise`, `RicianNoise`, `GammaCorrection`, `BiasField`, `WindowLevel`, `BrightnessContrast`, `GaussianBlur`, `MedianBlur`, `SimulateLowResolution`, `Sharpen`, `CLAHEContrast`, `HistogramMatch` |
| `modality/mri/` | MRI-specific artifacts | `GhostingArtifact`, `KSpaceDropout`, `MRIMotion` |
| `modality/ct/` | CT-specific artifacts | `BeamHardening`, `MetalStreak` |
| `modality/xray/` | X-ray (DXR) artifacts | `ScatterSimulation`, `GridArtifact` |
| `modality/tomosynthesis/` | DBT-specific | `SlabShift`, `LimitedAngleBlur`, `SliceDropout`, `AnisotropicElastic`, `CompressionVariation`, `ReconStreak` |

### Mask consistency contract

Every spatial transform must:

1. Apply the *same* sampled parameters to the image and the mask within a
   single call.
2. Use **nearest-neighbour interpolation** (`order=0`) for the mask.
3. Preserve the mask's dtype.

Intensity and modality transforms must not modify the mask at all (with the
one opt-in exception of `SliceDropout(affect_mask=True)`).

This is what makes `Compose([RandomAffine(...), ElasticDeform(...)])` safe
to use for segmentation training. A 100-seed regression test covering
image/mask alignment lives in `tests/integration/test_full_pipeline.py`.

### Anisotropic awareness

DBT volumes typically have `(1.0, 0.1, 0.1) mm` spacing — ten times more
compressed along Z than in-plane. Elastic deformation, bias fields, and
limited-angle blur are all anisotropy-aware:

- `ElasticDeform(alpha=(120, 120, 10), sigma=(10, 10, 3))` — per-axis magnitude
- `AnisotropicElastic` — thin wrapper with DBT defaults that fails fast on 2D input
- `LimitedAngleBlur` — applies blur only along Z, scaled by acquisition arc angle
- `BiasField(coarse_shape=4)` — independent coarse grid per axis, upsampled uniformly

### Serialisation contract

Every transform must override `to_dict()` to return a dict whose `"params"`
can be passed as keyword arguments to `__init__()` to reconstruct it. New
transforms should design their stored attributes to match their `__init__`
parameter names, or explicitly map them in `to_dict()`.

---

## 3. Serialisation (`medaugmentx/serialization.py`)

Enables lossless round-trip persistence of any pipeline.

```
pipeline  ──to_json()──►  JSON string  ──from_json()──►  pipeline
pipeline  ──to_yaml()──►  YAML string  ──from_yaml()──►  pipeline  (requires PyYAML)
```

**`REGISTRY`** maps class names to classes. All 36 built-in transforms are
registered at import time. Custom transforms can be added with the
`@register_transform` decorator (validates the class and prevents accidental
name collisions) or by direct assignment:

```python
from medaugmentx.serialization import register_transform

@register_transform
class MyTransform(Transform):
    ...
```

**`from_dict(d)`** reconstructs any transform from its dict form.
Containers (`Compose`, `OneOf`, `SomeOf`) are handled recursively — the
`"transforms"` list in `params` is reconstructed before the container itself.

JSON serialisation uses only the Python standard library. YAML is optional
(`pip install pyyaml`) and exposed via `to_yaml` / `from_yaml`.

---

## 4. Presets (`medaugmentx/presets.py`)

Four factory functions return fully-configured, seeded `Compose` pipelines:

| Function | Modality | Key transforms |
| --- | --- | --- |
| `mri_pipeline(seed)` | MRI | affine, elastic, bias field, Rician noise, ghosting or k-space dropout |
| `ct_pipeline(seed)` | CT | affine, elastic, window/level, Gaussian noise, beam hardening |
| `dxr_pipeline(seed)` | Digital X-ray | affine, blur, brightness/contrast, gamma, low-resolution sim |
| `dbt_pipeline(seed)` | DBT | affine, anisotropic elastic, slab shift, limited-angle blur, slice dropout, bias field |

All presets are serialisable via `to_json` / `to_yaml`.

---

## 5. Interop layer (`medaugmentx/interop/`)

Framework adapters sit above the core transform API. They make the library
usable in PyTorch, torchvision, and MONAI-style training code without making
those frameworks hard dependencies. TorchIO subjects are supported through the
same import-lazy adapter layer.

| Adapter | Purpose |
| --- | --- |
| `SampleTransform` | Generic adapter for `MedVolume`, arrays/tensors, `(image, mask)`, and dict samples |
| `TorchTransform` | PyTorch / torchvision-friendly alias of `SampleTransform` |
| `MonaiMapTransform` | Dict adapter with `image` / `label` defaults |
| `TorchIOTransform` | TorchIO `Subject` adapter for one image plus one optional label map |

Import rule: `medaugmentx.interop` must not import PyTorch, MONAI, or TorchIO
at module import time. Tensor and subject support is duck-typed and restores
framework objects only when those objects are actually passed.

Channel rule: MedAugmentX transforms operate on single-channel 2D/3D images.
Adapters may strip one singleton channel axis before augmentation and restore
it after augmentation.

---

## 6. I/O layer (`medaugmentx/io/`)

Loaders are **optional dependencies**. Top-level imports do not require
`pydicom` or `nibabel`; the helpers raise a clear `ImportError` only when
called without their backend installed.

| Helper | Backend | Returns |
| --- | --- | --- |
| `load_dicom_series(path)` | `pydicom` | `MedVolume` (3D from series, 2D from single file) |
| `load_nifti(path)` | `nibabel` | `MedVolume` |
| `save_nifti(vol, path)` | `nibabel` | None |

DICOM loading sorts slices by image position projected onto the slice normal,
applies `RescaleSlope * pixels + RescaleIntercept`, and reads spacing from
`PixelSpacing` plus the median inter-slice distance. Vendor metadata
(`Manufacturer`, `SeriesInstanceUID`, etc.) is preserved in `metadata`.

---

## 7. Dependencies

| Layer | Required | Optional |
| --- | --- | --- |
| Core | `numpy` | — |
| Transforms | `scipy` | — |
| Serialisation | — | `pyyaml` (YAML only; JSON uses stdlib) |
| I/O | — | `pydicom`, `nibabel` |
| Interop | — | `torch`, `monai`, `torchio` |

Phase 3 framework support stays behind extras so installing MedAugmentX never
pulls in a deep-learning framework by default.

---

## 8. Commercial adoption posture

MedAugmentX is intentionally local, serialisable, and dependency-light:

- no runtime network access,
- no telemetry,
- optional I/O and framework integrations,
- JSON/YAML pipeline configs for audit trails,
- typed public package surface through `py.typed`,
- stable axis and mask-consistency contracts.

The library is intended for medical AI training and evaluation workflows. It is
not a diagnostic device, clinical decision-support system, or substitute for
local clinical, security, privacy, and regulatory validation. See
[COMMERCIAL_ADOPTION.md](COMMERCIAL_ADOPTION.md) and
[SECURITY.md](../SECURITY.md).

---

## 9. Testing strategy

- **Unit tests** for every transform — output shape, dtype, value range,
  reproducibility under a fixed seed, `p=0` boundary, mask untouched.
- **Mask consistency** regression tests for spatial transforms.
- **Integration tests** under `tests/integration/` exercise the full pipeline
  end-to-end with a 100-seed regression for image/mask alignment.
- **Serialisation tests** verify JSON round-trip for all 36 registered
  transforms and all three container types.
- **Preset tests** verify that each preset runs on the correct volume
  dimensionality, produces bit-identical output under the same seed, and
  round-trips through serialisation.
- **I/O tests** are marked `@pytest.mark.io` and skipped automatically when
  the optional dependency is missing.
- **Interop tests** verify dict, tuple/list, channel-restoration, and
  MONAI-style label-key behavior without requiring PyTorch to be installed.

---

## 10. Public surface

The flat re-export in `medaugmentx/transforms/__init__.py` is the canonical
import path. Internal modules (`medaugmentx/transforms/intensity/bias_field.py`,
etc.) may move between minor versions; the public surface follows SemVer once
we hit `1.0`. The developer-facing reference lives in
[API_REFERENCE.md](API_REFERENCE.md).

---

## 11. Validation & guards (`medaugmentx/validation.py`)

This layer sits above the transforms and depends only on `core` and NumPy. It
exists because a transform that emits `NaN`, collapses the dynamic range,
desynchronises the mask, or crops away the only labelled structure does not
raise — it silently poisons the dataset.

`VolumeValidator` runs a configurable set of plausibility rules over a
`MedVolume`, splitting findings into error- and warning-severity issues
collected in a `ValidationReport`. Rules that need the pre-augmentation volume
(label preservation, foreground loss, intensity drift) only fire when a
`reference` is supplied.

`Guard` is a `Transform` that wraps another transform (or a whole pipeline),
validates its output against the input on every call, and reacts per
`on_fail`: `raise`, `warn`, `revert` to the input, or `retry` with a fresh
derived RNG stream. Because it is a `Transform`, it nests in the pipeline
containers and round-trips through the serialisation `REGISTRY` — `from_dict`
rebuilds its single wrapped transform and reconstructs the validator from its
plain-dict config.
