# Architecture

MedAugmentX is a layered library. Each layer has one responsibility, no
upward dependencies, and a flat public surface. This document explains the
layout and the rules each layer follows.

```
┌─────────────────────────────────────────────────────────────────┐
│ Policy Studio  (medaugmentx/studio/, `medaugmentx studio`)       │
│   catalog · session · render · stdlib HTTP server · static UI   │
├─────────────────────────────────────────────────────────────────┤
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
    # Optional geometric targets, appended so the original positional
    # signature (image, mask, spacing, metadata) is preserved:
    keypoints: Optional[np.ndarray]  # (N, ndim) coords, array-index order
    keypoint_labels: Optional[np.ndarray]  # (N,) parallel labels
    bboxes: Optional[np.ndarray]     # (M, 2*ndim) = [min…, max…], array order
    bbox_labels: Optional[np.ndarray]  # (M,) parallel labels
```

Invariants enforced in `__post_init__`:

- image is 2D or 3D,
- mask, when present, has the same shape as image,
- keypoints are `(N, ndim)`, bboxes are `(M, 2*ndim)` with `min <= max`,
- labels, when present, are 1D with one entry per target,
- spacing length matches image ndim (defaults to all-1.0 if omitted),
- spatial axes are non-empty; spacing is finite and positive; target coordinates are finite,
- metadata is a dict,
- affine, when known, maps array indices to RAS mm and agrees with spacing.

`replace()` returns a shallow copy with selected fields swapped (`None` means
"keep current", so an intensity transform that only rewrites the image carries
mask and targets through unchanged). `copy()` deep-copies the arrays. `warp()`
swaps in an already-transformed image while mapping targets through a coordinate
function, and `remove_out_of_bounds_targets()` prunes off-frame targets.

### Geometric targets (`core/geometry.py`)

Keypoints and bounding boxes are optional, stored in **array-index order**
(same as the image axes). `core/geometry.py` is a pure-NumPy module of *point
maps* — `flip_map`, `affine_map`, `translate_map`, `scale_map`,
`displacement_map` — plus `map_keypoints`/`map_bboxes`. A spatial transform
expresses its pixel warp as one of these maps and hands it to `MedVolume.warp`,
which applies it to keypoints and (via their `2**ndim` corners, re-bounded to a
valid axis-aligned box) to bounding boxes. Labels ride along untouched.
Intensity, artifact, and occlusion (`CoarseDropout`) transforms leave geometry —
and therefore targets — unchanged. Targets are never dropped mid-transform;
pruning is an explicit post-step.

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
  reproduces a sequence of outputs in an equivalent software environment.
  Calls advance the state; inserting or reordering children can change their
  assigned streams.
- **`OneOf`** picks exactly one child, optionally weighted, and forces it to
  run regardless of the child's own `p`.
- **`SomeOf`** picks `n` (or a range) children without replacement and
  applies them in deterministic order.
- **`PerChannel`** wraps one intensity transform and applies it to the primary
  image and the selected named channels, with independent draws from one
  derived stream or, with `shared=True`, by restoring every RNG state in the
  wrapped subtree before each image. It rejects children that change shape,
  mask, geometry, or targets.

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
| `spatial/` | Geometry — pixels move | `RandomAffine`, `RandomFlip`, `AnatomicCrop`, `ClassBalancedCrop`, `ElasticDeform`, `CoarseDropout`, `Resize`, `ResampleToSpacing`, `ResampleToGrid`, `Pad`, `CenterCrop` |
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

DBT volumes can have `(1.0, 0.1, 0.1) mm` spacing — coarser sampling along Z
than in-plane. Strength must match the task/grid. Affine angles use the physical metric;
elastic and Gaussian blur parameters support explicit voxel/mm units:

- `ElasticDeform(alpha=(10, 120, 120), sigma=(3, 10, 10))` — per-axis magnitude
- `AnisotropicElastic` — thin wrapper with DBT defaults that fails fast on 2D input
- `RandomAffine(axes_enabled=("z",))` — rotation about the slice axis, in-plane only
- `LimitedAngleBlur` — applies blur only along Z, scaled by acquisition arc angle
- `BiasField(coarse_shape=4)` — independent coarse grid per axis, upsampled uniformly

Every displacement component uses the same spatial `sigma` tuple; `alpha`
controls the component magnitudes. Landmark mapping solves the inverse pull
field to a bounded residual and raises if it cannot converge. Nonlinear box
mapping defaults to conservative global displacement bounds; optional corner
mapping is an approximation.

### Serialisation contract

Every transform must override `to_dict()` to return a dict whose `"params"`
can be passed as keyword arguments to `__init__()` to reconstruct it. New
transforms should design their stored attributes to match their `__init__`
parameter names, or explicitly map them in `to_dict()`.

---

## 3. Serialisation (`medaugmentx/serialization.py`)

Persists constructor policies with schema/library versions and a canonical hash.
Current RNG checkpoints are stored separately by record_augmentation.

```
pipeline  ──to_json()──►  JSON string  ──from_json()──►  pipeline
pipeline  ──to_yaml()──►  YAML string  ──from_yaml()──►  pipeline  (requires PyYAML)
```

**`REGISTRY`** maps class names to classes. All built-in transforms and containers are
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
The single-child wrappers listed in `WRAPPER_NAMES` (`Guard`, `PerChannel`)
nest their child under `params["transform"]` and are rebuilt the same way.

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

Channel rule: adapters accept single-channel 2D/3D image fields. For several
registered images, construct a `MedVolume` with named channels, wrap intensity
policies in `PerChannel`, and use `stack_channels` after augmentation.
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
we hit `1.0`. What that surface covers, and the semantics shared across it, are
in [Public surface & semantics](API_REFERENCE.md); the generated
[transform catalog](sphinx/catalog.rst) and [API reference](sphinx/api/index.rst)
carry the signatures.

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

---

## 12. Policy Studio (`medaugmentx/studio/`)

The studio is an application layer on top of every other layer; nothing below
imports it. It is split so each part can be tested without a browser:

- `catalog.py` derives the editor's metadata from `REGISTRY`, constructor
  signatures, `Args:` docstring sections, and the packaged model cards. A
  registered transform appears in the studio with no further wiring, so a
  missing parameter description is caught by the studio tests.
- `session.py` (`StudioSession`) owns the loaded volume and the edited policy in
  its canonical `to_dict()` form. Previews reseed a compiled copy with
  `seed_for_sample`, exactly like `apply_for_sample`, and record which steps ran
  through the `ACTIVE_TRACE` context variable used by `record_augmentation`.
  The last draw is cached by policy, draw options, and a volume version so
  slice and display changes do not recompute the augmentation.
- `render.py` slices, windows, and overlays labels, and encodes PNG with
  `zlib` alone, so no plotting dependency is needed.
- `server.py` is a `ThreadingHTTPServer` with a JSON API. It binds to loopback,
  requires a per-session token, checks Host and Origin headers, streams uploads
  to a private temporary folder, and serves a strict Content-Security-Policy.
- `static/` holds a dependency-free HTML/CSS/JS front-end packaged as
  `package-data`; it never fetches remote assets.

`medaugmentx/cli.py` provides the `medaugmentx studio` console script, and
`python -m medaugmentx.studio` runs the same command.
