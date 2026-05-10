# Architecture

MedAugment is a layered library. Each layer has one responsibility, no
upward dependencies, and a flat public surface. This document explains the
layout and the rules each layer follows.

```
┌─────────────────────────────────────────────────────────────────┐
│ Presets  (medaugment/presets.py)                                │
│   mri_pipeline · ct_pipeline · dxr_pipeline · dbt_pipeline     │
├─────────────────────────────────────────────────────────────────┤
│ Serialisation  (medaugment/serialization.py)                    │
│   REGISTRY · from_dict · to_json / from_json · to_yaml / from_yaml │
├─────────────────────────────────────────────────────────────────┤
│ User pipeline  (Compose · OneOf · SomeOf)                       │
├─────────────────────────────────────────────────────────────────┤
│ Transforms                                                      │
│   spatial/   intensity/   modality/mri/   modality/ct/          │
│                           modality/tomosynthesis/               │
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
```

---

## 1. Core layer (`medaugment/core/`)

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
  `medaugment.serialization.from_dict()`.

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

## 2. Transforms layer (`medaugment/transforms/`)

Subdivided by *what* changes about the image:

| Folder | Changes | Contents |
| --- | --- | --- |
| `spatial/` | Geometry — pixels move | `RandomAffine`, `RandomFlip`, `AnatomicCrop`, `ElasticDeform` |
| `intensity/` | Per-pixel value, geometry preserved | `GaussianNoise`, `RicianNoise`, `GammaCorrection`, `BiasField`, `WindowLevel`, `BrightnessContrast`, `GaussianBlur`, `SimulateLowResolution` |
| `modality/mri/` | MRI-specific artifacts | `GhostingArtifact`, `KSpaceDropout` |
| `modality/ct/` | CT-specific artifacts | `BeamHardening` |
| `modality/tomosynthesis/` | DBT-specific | `SlabShift`, `LimitedAngleBlur`, `SliceDropout`, `AnisotropicElastic` |

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

## 3. Serialisation (`medaugment/serialization.py`)

Enables lossless round-trip persistence of any pipeline.

```
pipeline  ──to_json()──►  JSON string  ──from_json()──►  pipeline
pipeline  ──to_yaml()──►  YAML string  ──from_yaml()──►  pipeline  (requires PyYAML)
```

**`REGISTRY`** maps class names to classes. All 22 built-in transforms are
registered at import time. Custom transforms can be added:

```python
from medaugment.serialization import REGISTRY
REGISTRY["MyTransform"] = MyTransform
```

**`from_dict(d)`** reconstructs any transform from its dict form.
Containers (`Compose`, `OneOf`, `SomeOf`) are handled recursively — the
`"transforms"` list in `params` is reconstructed before the container itself.

JSON serialisation uses only the Python standard library. YAML is optional
(`pip install pyyaml`) and exposed via `to_yaml` / `from_yaml`.

---

## 4. Presets (`medaugment/presets.py`)

Four factory functions return fully-configured, seeded `Compose` pipelines:

| Function | Modality | Key transforms |
| --- | --- | --- |
| `mri_pipeline(seed)` | MRI | affine, elastic, bias field, Rician noise, ghosting or k-space dropout |
| `ct_pipeline(seed)` | CT | affine, elastic, window/level, Gaussian noise, beam hardening |
| `dxr_pipeline(seed)` | Digital X-ray | affine, blur, brightness/contrast, gamma, low-resolution sim |
| `dbt_pipeline(seed)` | DBT | affine, anisotropic elastic, slab shift, limited-angle blur, slice dropout, bias field |

All presets are serialisable via `to_json` / `to_yaml`.

---

## 5. I/O layer (`medaugment/io/`)

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

## 6. Dependencies

| Layer | Required | Optional |
| --- | --- | --- |
| Core | `numpy` | — |
| Transforms | `scipy` | — |
| Serialisation | — | `pyyaml` (YAML only; JSON uses stdlib) |
| I/O | — | `pydicom`, `nibabel` |

Phase 3 adds an optional PyTorch backend for GPU-accelerated spatial
transforms; it stays behind an extra so installing MedAugment never pulls
in a deep-learning framework by default.

---

## 7. Testing strategy

- **Unit tests** for every transform — output shape, dtype, value range,
  reproducibility under a fixed seed, `p=0` boundary, mask untouched.
- **Mask consistency** regression tests for spatial transforms.
- **Integration tests** under `tests/integration/` exercise the full pipeline
  end-to-end with a 100-seed regression for image/mask alignment.
- **Serialisation tests** verify JSON round-trip for all 22 registered
  transforms and all three container types.
- **Preset tests** verify that each preset runs on the correct volume
  dimensionality, produces bit-identical output under the same seed, and
  round-trips through serialisation.
- **I/O tests** are marked `@pytest.mark.io` and skipped automatically when
  the optional dependency is missing.

---

## 8. Public surface

The flat re-export in `medaugment/transforms/__init__.py` is the canonical
import path. Internal modules (`medaugment/transforms/intensity/bias_field.py`,
etc.) may move between minor versions; the public surface follows SemVer once
we hit `1.0`.
