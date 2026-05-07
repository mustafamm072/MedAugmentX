# Architecture

MedAugment is a layered library. Each layer has one responsibility, no
upward dependencies, and a flat public surface. This document explains the
layout and the rules each layer follows.

```
┌────────────────────────────────────────────────────────────┐
│ User pipeline (Compose, OneOf, SomeOf)                     │
├────────────────────────────────────────────────────────────┤
│ Transforms                                                 │
│   spatial/   intensity/   modality/tomosynthesis/          │
├────────────────────────────────────────────────────────────┤
│ Core                                                       │
│   MedVolume · Transform ABC · seedable RNG · helpers       │
├────────────────────────────────────────────────────────────┤
│ I/O                                                        │
│   DICOM series · NIfTI                                     │
└────────────────────────────────────────────────────────────┘
              numpy + scipy (always)
              pydicom + nibabel (optional, behind extras)
```

## 1. Core layer (`medaugment/core/`)

The foundational types. **Nothing else in the library is allowed without it,
and it has no dependencies of its own beyond NumPy.**

### `MedVolume` (`core/volume.py`)

```python
@dataclass
class MedVolume:
    image: np.ndarray                 # (D, H, W) or (H, W), recommended float32
    mask: Optional[np.ndarray]        # same shape as image, integer labels
    spacing: Tuple[float, ...]        # mm per axis, one entry per ndim
    metadata: Dict[str, Any]          # modality, vendor, original tags...
```

Invariants enforced in `__post_init__`:

- image is 2D or 3D,
- mask, when present, has the same shape as image,
- spacing length matches image ndim (defaults to all-1.0 if omitted),
- metadata is a dict.

`replace()` returns a shallow-copy with selected fields swapped. `copy()`
deep-copies the arrays.

### Axis convention

| Volume | Storage order | Axis 0 | Axis 1 | Axis 2 |
| --- | --- | --- | --- | --- |
| 3D | `(D, H, W)` | z (slice) | y (row) | x (col) |
| 2D | `(H, W)` | y (row) | x (col) | — |

`spacing` follows the same order. The DICOM and NIfTI loaders both transpose
to this convention so that downstream code never has to think about it.

### `Transform` (`core/base.py`)

Abstract base class. Subclasses override `apply(volume) -> MedVolume`. The
base class handles:

- probability gating (`p` in `[0, 1]`),
- the `self.rng` generator (always used in transforms, never `np.random`),
- a `to_dict()` introspection helper (full YAML serialisation lands in Phase 2).

### `Compose`, `OneOf`, `SomeOf` (`core/compose.py`)

- **`Compose`** runs children sequentially. The top-level seed is *spawned*
  into one independent generator per child via `derive_rng`, so a given seed
  produces bit-identical output every time. Adding or removing transforms
  does not change the seed assignments of the unchanged ones.
- **`OneOf`** picks exactly one child, optionally weighted, and forces it to
  run regardless of the child's own `p`.
- **`SomeOf`** picks `n` (or a range) children without replacement and
  applies them in deterministic order.

### Seeding rules

1. Every transform owns a `np.random.Generator` (`self.rng`).
2. `Compose` derives child seeds from its own seed; passing a seed to the
   top level is sufficient for end-to-end reproducibility.
3. **Transforms must never call `np.random` directly.** Use `self.rng`. CI
   does not (yet) lint for this; reviewers should.

## 2. Transforms layer (`medaugment/transforms/`)

Subdivided by *what* changes about the image:

| Folder | Changes | Phase 1 contents |
| --- | --- | --- |
| `spatial/` | Geometry — pixels move | `RandomAffine`, `RandomFlip`, `AnatomicCrop`, `ElasticDeform` |
| `intensity/` | Per-pixel value, geometry preserved | `GaussianNoise`, `RicianNoise`, `GammaCorrection` |
| `modality/<modality>/` | Anything that only makes sense for one modality | `tomosynthesis/`: `SlabShift`, `LimitedAngleBlur`, `SliceDropout`, `AnisotropicElastic` |

### Mask consistency contract

Every spatial transform must:

1. Apply the *same* sampled parameters to the image and the mask within a
   single call.
2. Use **nearest-neighbour interpolation** (`order=0`) for the mask.
3. Preserve the mask's dtype.

This is what makes `Compose([RandomAffine(...), ElasticDeform(...)])` safe
to use for segmentation training. There is a regression test covering 100
random seeds in `tests/integration/test_full_pipeline.py`.

### Anisotropic awareness

Phase 1 builds the primitives that Phase 2 will compose into modality
presets. The most important is anisotropic-aware sigma/alpha for elastic
deformation: a DBT volume with `(1.0, 0.1, 0.1) mm` spacing must not be
warped along Z as much as in-plane, so `ElasticDeform(alpha=(120, 120, 10),
sigma=(10, 10, 3))` is the canonical setup. `AnisotropicElastic` is a thin
wrapper with DBT defaults that fails fast on 2D input.

## 3. I/O layer (`medaugment/io/`)

Loaders are **optional dependencies**. The top-level imports do not require
`pydicom` or `nibabel`; the helpers raise a clear `ImportError` only when
called without their backend installed.

| Helper | Backend | Returns |
| --- | --- | --- |
| `load_dicom_series(path)` | `pydicom` | `MedVolume` (3D from a series, 2D from a single file) |
| `load_nifti(path)` | `nibabel` | `MedVolume` |
| `save_nifti(vol, path)` | `nibabel` | None |

DICOM loading sorts slices by image position projected onto the slice
normal, applies `RescaleSlope * pixels + RescaleIntercept`, and reads
spacing from `PixelSpacing` plus the median inter-slice distance. Vendor
metadata (`Manufacturer`, `SeriesInstanceUID`, etc.) is preserved.

Phase 2 adds vendor-specific multi-frame DBT parsers (Hologic Selenia
Dimensions, GE SenoClaire, Siemens MAMMOMAT Revelation).

## 4. Tomosynthesis as a first-class modality

DBT is the third major axis of design (alongside MRI and CT) because no
existing library supports it natively. Phase 1 ships the four most-needed
DBT transforms:

| Transform | Effect |
| --- | --- |
| `SlabShift` | Z-axis recon-centre variation (typical: ±2 slices) |
| `LimitedAngleBlur` | Z-only Gaussian blur scaled by acquisition arc |
| `SliceDropout` | Zero a small number of slices (robustness) |
| `AnisotropicElastic` | DBT-default `alpha=(100,100,8)`, `sigma=(8,8,2)` |

A full `TOMOSYNTHESIS_STANDARD` preset, vendor parsers, and the
remaining DBT transforms (`CompressionVariation`, `ReconStreak`) arrive in
Phase 2.

## 5. Dependencies

| Layer | Required | Optional |
| --- | --- | --- |
| Core | `numpy` | — |
| Transforms | `scipy.ndimage` | — |
| I/O | — | `pydicom`, `nibabel` |

Phase 3 adds an optional PyTorch backend for GPU-accelerated spatial
transforms; it stays behind an extra so installing MedAugment never pulls
in a deep-learning framework by default.

## 6. Testing strategy

- **Unit tests** for every transform — output shape, dtype, value range,
  reproducibility under a fixed seed, `p=0` boundary.
- **Mask consistency** tests for spatial transforms.
- **Integration tests** under `tests/integration/` exercise the full
  pipeline end-to-end and include a 100-seed regression test for image/mask
  alignment.
- **I/O tests** are marked `@pytest.mark.io` and skipped automatically when
  the optional dependency is missing.

## 7. Public surface

The flat re-export in `medaugment/__init__.py` is the only thing users are
expected to import from. Internal modules may move between minor versions;
the public surface follows SemVer once we hit `1.0`.
