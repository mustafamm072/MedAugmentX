# API Reference

Version: `0.7.0`

This page documents the supported public imports. Prefer these paths in
applications, papers, tutorials, and commercial code; internal module paths may
move before `1.0`.

---

## Core API

```python
from medaugmentx import (
    MedVolume,
    Transform,
    Compose,
    OneOf,
    SomeOf,
    PipelineStep,
    iter_pipeline,
    pipeline_summary,
)
```

### `MedVolume`

```python
MedVolume(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    spacing: tuple[float, ...] = (),
    metadata: dict[str, Any] = {},
)
```

Container for one 2D `(H, W)` or 3D `(D, H, W)` medical image. `mask`, when
provided, must match `image.shape`. `spacing` uses the same axis order as the
image and defaults to `1.0` per axis.

Useful properties and methods:

| API | Description |
| --- | --- |
| `volume.ndim` | `2` or `3` |
| `volume.shape` | Image shape tuple |
| `volume.is_3d` | True for `(D, H, W)` volumes |
| `volume.has_mask` | True when a mask is attached |
| `volume.modality` | `metadata.get("modality")` |
| `volume.replace(...)` | Return a copy with selected fields changed |
| `volume.copy()` | Deep-copy arrays and metadata |

### `Transform`

```python
Transform(p: float = 1.0, seed: int | np.random.Generator | None = None)
```

Base class for all augmentations. Subclasses implement `apply(volume)` and use
`self.rng` for all random sampling. Calling a transform validates that the
input is a `MedVolume`, applies probability gating through `p`, and returns a
new `MedVolume`.

### Pipeline containers

```python
Compose(transforms, p=1.0, seed=None)
OneOf(transforms, weights=None, p=1.0, seed=None)
SomeOf(transforms, n=1, p=1.0, seed=None)
```

`Compose` applies children sequentially. `OneOf` chooses exactly one child.
`SomeOf` chooses `n` children without replacement. All containers seed child
transforms deterministically from the parent seed.

### Pipeline inspection

```python
pipeline_summary(transform, max_value_length=72) -> str
iter_pipeline(transform) -> Iterator[PipelineStep]
```

`pipeline_summary` returns a compact multi-line tree for a single transform or
nested pipeline. It is intended for experiment logs, review notes, and saved
augmentation policy summaries.

`iter_pipeline` yields `PipelineStep` objects in depth-first order:

| Field | Description |
| --- | --- |
| `path` | Tuple of child indices from the root; root is `()` |
| `name` | Transform class name |
| `params` | Parameters from `to_dict()`, excluding nested `transforms` |
| `depth` | Nesting depth, equal to `len(path)` |

---

## Transform API

Use the flat transform namespace:

```python
from medaugmentx.transforms import RandomAffine, RicianNoise, BiasField
```

### Spatial

| Transform | Constructor summary |
| --- | --- |
| `RandomAffine` | `rotation=0.0, scale=(1.0, 1.0), translation=(0.0, 0.0), axes_enabled=("x", "y", "z"), order=1, mode="constant", cval=0.0, p=1.0, seed=None` |
| `RandomFlip` | `axes=("x",), p_per_axis=0.5, p=1.0, seed=None` |
| `AnatomicCrop` | `size, foreground_prob=0.5, foreground_threshold=0.0, p=1.0, seed=None` |
| `ElasticDeform` | `alpha=30.0, sigma=4.0, order=1, mode="reflect", cval=0.0, p=1.0, seed=None` |
| `CoarseDropout` | `num_holes=(1, 4), hole_size=(0.05, 0.2), fill_value=0.0, fill_mask=False, p=1.0, seed=None` |
| `Resize` | `size, order=1, p=1.0, seed=None` |
| `Pad` | `size, mode="constant", cval=0.0, p=1.0, seed=None` |
| `CenterCrop` | `size, p=1.0, seed=None` |

Spatial transforms apply the same sampled geometry to `image` and `mask`, and
use nearest-neighbour interpolation for masks. `Resize`/`Pad`/`CenterCrop` are
deterministic shape-normalisation helpers (`Pad` never crops, `CenterCrop`
never pads — pair them to force an exact shape).

### Intensity

| Transform | Constructor summary |
| --- | --- |
| `GaussianNoise` | `std=0.01, relative=False, clip=None, p=1.0, seed=None` |
| `RicianNoise` | `std=0.01, clip=None, p=1.0, seed=None` |
| `GammaCorrection` | `gamma=(0.8, 1.2), invert=False, p=1.0, seed=None` |
| `BiasField` | `alpha=0.3, coarse_shape=4, order=3, p=1.0, seed=None` |
| `WindowLevel` | `center_shift_frac=0.1, width_scale=(0.8, 1.2), rescale_output=True, p=1.0, seed=None` |
| `BrightnessContrast` | `brightness=0.0, contrast=(0.9, 1.1), clip=None, p=1.0, seed=None` |
| `GaussianBlur` | `sigma=(0.5, 1.5), order=0, mode="reflect", p=1.0, seed=None` |
| `MedianBlur` | `ksize=3, mode="reflect", p=1.0, seed=None` |
| `SimulateLowResolution` | `zoom_range=(0.5, 0.9), order_down=1, order_up=1, per_axis=False, p=1.0, seed=None` |
| `Sharpen` | `alpha=(0.2, 0.8), sigma=(0.7, 1.5), clip=None, p=1.0, seed=None` |
| `CLAHEContrast` | `clip_limit=(1.0, 3.0), grid=(8, 8), n_bins=256, p=1.0, seed=None` |
| `HistogramMatch` | `reference=None, blend=1.0, n_quantiles=256, p=1.0, seed=None` |

Intensity transforms leave masks unchanged.

### Modality-Specific

| Modality | Transform | Constructor summary |
| --- | --- | --- |
| MRI | `GhostingArtifact` | `ghost_intensity=(0.05, 0.15), ghost_shift=(8, 32), phase_encode_axis="y", num_ghosts=1, p=1.0, seed=None` |
| MRI | `KSpaceDropout` | `dropout_fraction=(0.01, 0.05), phase_encode_axis="y", p=1.0, seed=None` |
| MRI | `MRIMotion` | `degrees=(1.0, 5.0), translation=(1.0, 4.0), num_movements=(1, 3), p=1.0, seed=None` |
| CT | `BeamHardening` | `alpha=(0.02, 0.08), power=2.0, p=1.0, seed=None` |
| CT | `MetalStreak` | `intensity=(0.1, 0.3), num_streaks=(6, 12), num_sources=1, falloff=0.5, p=1.0, seed=None` |
| X-ray | `ScatterSimulation` | `fraction=(0.1, 0.4), sigma=(15.0, 40.0), p=1.0, seed=None` |
| X-ray | `GridArtifact` | `amplitude=(0.03, 0.1), frequency=(0.2, 0.45), axis="x", p=1.0, seed=None` |
| DBT | `SlabShift` | `max_shift=2, cval=0.0, p=1.0, seed=None` |
| DBT | `LimitedAngleBlur` | `arc_degrees=(15.0, 25.0), base_sigma=1.0, reference_arc_deg=20.0, p=1.0, seed=None` |
| DBT | `SliceDropout` | `num_slices=1, cval=0.0, affect_mask=False, p=1.0, seed=None` |
| DBT | `AnisotropicElastic` | `alpha=(100.0, 100.0, 8.0), sigma=(8.0, 8.0, 2.0), order=1, p=1.0, seed=None` |
| DBT | `CompressionVariation` | `scale=(0.85, 1.15), axis="y", order=1, p=1.0, seed=None` |
| DBT | `ReconStreak` | `amplitude=(0.05, 0.2), num_planes=(1, 3), displacement=1.5, decay=0.6, axis="x", p=1.0, seed=None` |

---

## Presets

```python
from medaugmentx.presets import mri_pipeline, ct_pipeline, dxr_pipeline, dbt_pipeline
```

| Factory | Returns |
| --- | --- |
| `mri_pipeline(seed=None)` | MRI spatial, bias-field, Rician noise, ghosting/k-space pipeline |
| `ct_pipeline(seed=None)` | CT spatial, window/level, noise, beam-hardening pipeline |
| `dxr_pipeline(seed=None)` | 2D digital X-ray pipeline |
| `dbt_pipeline(seed=None)` | Digital breast tomosynthesis pipeline |

Each preset returns a serialisable `Compose`.

---

## Serialisation

```python
from medaugmentx.serialization import REGISTRY, from_dict, to_json, from_json, to_yaml, from_yaml
```

| API | Description |
| --- | --- |
| `REGISTRY` | Maps transform class names to classes |
| `register_transform(cls=None, *, name=None, override=False)` | Decorator that registers a custom transform class |
| `from_dict(d)` | Reconstruct a transform from `transform.to_dict()` |
| `to_json(transform, indent=2)` | Serialise a transform or pipeline to JSON |
| `from_json(s)` | Reconstruct from JSON |
| `to_yaml(transform)` | Serialise to YAML; requires `pyyaml` |
| `from_yaml(s)` | Reconstruct from YAML; requires `pyyaml` |

Custom transform classes can be registered with the `@register_transform`
decorator (which validates the class and guards against name collisions) or by
direct assignment, `REGISTRY["MyTransform"] = MyTransform`.

---

## Framework Interop

```python
from medaugmentx.interop import (
    SampleTransform,
    TorchTransform,
    MonaiMapTransform,
    TorchIOTransform,
)
```

### `SampleTransform`

```python
SampleTransform(
    transform,
    image_key="image",
    mask_key="mask",
    spacing_key="spacing",
    metadata_key="metadata",
    channel_dim="auto",
    preserve_image_dtype=False,
    preserve_mask_dtype=True,
)
```

Adapts any MedAugmentX `Transform` or `Compose` to common dataset sample
formats:

| Input | Output |
| --- | --- |
| `MedVolume` | `MedVolume` |
| `np.ndarray` or `torch.Tensor` image | same array/tensor kind |
| `(image, mask)` tuple/list | same tuple/list kind |
| mapping with `image_key` and optional `mask_key` | shallow-copied mapping |

`channel_dim` may be `None`, `"auto"`, `"first"`, `"last"`, or an integer
axis. MedAugmentX operates on single-channel 2D/3D data; singleton channel
dimensions are stripped before augmentation and restored after augmentation.

### `TorchTransform`

PyTorch/torchvision-friendly alias of `SampleTransform`. Importing it does not
import torch; tensor support is activated only when a torch tensor is passed.

### `MonaiMapTransform`

```python
MonaiMapTransform(
    transform,
    image_key="image",
    label_key="label",
    spacing_key="spacing",
    metadata_key="metadata",
    channel_dim="auto",
    preserve_image_dtype=False,
    preserve_label_dtype=True,
)
```

Dictionary adapter with MONAI-style `image` / `label` defaults.

### `TorchIOTransform`

```python
TorchIOTransform(
    transform,
    image_key="image",
    label_key="label",
    channel_dim=0,
    preserve_image_dtype=False,
    preserve_label_dtype=True,
    copy=True,
)
```

TorchIO `Subject` adapter. It reads one intensity image and one optional label
map from image-like objects with `.data`, runs the MedAugmentX transform, and
writes augmented data back to the returned subject.

`image_key` and `label_key` may be explicit strings. If `image_key` is absent
from the subject, the adapter infers a unique non-label image key. If
`label_key` is absent, it infers a unique label-like key when present. Pass
explicit keys for multi-image subjects.

Importing `TorchIOTransform` does not import TorchIO; install TorchIO only
when your pipeline needs it with `pip install "medaugmentx[torchio]"`.

---

## I/O

```python
from medaugmentx.io import load_dicom_series, load_nifti, save_nifti
```

| API | Extra | Description |
| --- | --- | --- |
| `load_dicom_series(path)` | `dicom` | Load a DICOM directory or file into `MedVolume` |
| `load_nifti(path)` | `nifti` | Load `.nii` / `.nii.gz` into `MedVolume` |
| `save_nifti(volume, path)` | `nifti` | Save a `MedVolume` to NIfTI |

Install optional I/O support with `pip install "medaugmentx[io]"`.

---

## Optional Extras

| Extra | Installs |
| --- | --- |
| `dicom` | `pydicom` |
| `nifti` | `nibabel` |
| `io` | `pydicom`, `nibabel` |
| `yaml` | `pyyaml` |
| `torch` | `torch` |
| `monai` | `monai` |
| `torchio` | `torchio` |
| `frameworks` | `torch`, `monai`, `torchio` |

---

## Operational Docs

| Document | Use |
| --- | --- |
| [Commercial adoption](COMMERCIAL_ADOPTION.md) | Intended use, validation, audit trail, privacy, and dependency guidance |
| [Architecture](ARCHITECTURE.md) | Layering, public surface, contracts, and dependency boundaries |
| [Security policy](../SECURITY.md) | Vulnerability reporting, PHI handling, supported versions, and dependency posture |
