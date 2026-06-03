# API Reference

Version: `0.3.0`

This page documents the supported public imports. Prefer these paths in
applications, papers, tutorials, and commercial code; internal module paths may
move before `1.0`.

---

## Core API

```python
from medaugmentx import MedVolume, Transform, Compose, OneOf, SomeOf
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

Spatial transforms apply the same sampled geometry to `image` and `mask`, and
use nearest-neighbour interpolation for masks.

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
| `SimulateLowResolution` | `zoom_range=(0.5, 0.9), order_down=1, order_up=1, per_axis=False, p=1.0, seed=None` |

Intensity transforms leave masks unchanged.

### Modality-Specific

| Modality | Transform | Constructor summary |
| --- | --- | --- |
| MRI | `GhostingArtifact` | `ghost_intensity=(0.05, 0.15), ghost_shift=(8, 32), phase_encode_axis="y", num_ghosts=1, p=1.0, seed=None` |
| MRI | `KSpaceDropout` | `dropout_fraction=(0.01, 0.05), phase_encode_axis="y", p=1.0, seed=None` |
| CT | `BeamHardening` | `alpha=(0.02, 0.08), power=2.0, p=1.0, seed=None` |
| DBT | `SlabShift` | `max_shift=2, cval=0.0, p=1.0, seed=None` |
| DBT | `LimitedAngleBlur` | `arc_degrees=(15.0, 25.0), base_sigma=1.0, reference_arc_deg=20.0, p=1.0, seed=None` |
| DBT | `SliceDropout` | `num_slices=1, cval=0.0, affect_mask=False, p=1.0, seed=None` |
| DBT | `AnisotropicElastic` | `alpha=(100.0, 100.0, 8.0), sigma=(8.0, 8.0, 2.0), order=1, p=1.0, seed=None` |

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
| `from_dict(d)` | Reconstruct a transform from `transform.to_dict()` |
| `to_json(transform, indent=2)` | Serialise a transform or pipeline to JSON |
| `from_json(s)` | Reconstruct from JSON |
| `to_yaml(transform)` | Serialise to YAML; requires `pyyaml` |
| `from_yaml(s)` | Reconstruct from YAML; requires `pyyaml` |

Custom transform classes can be registered with
`REGISTRY["MyTransform"] = MyTransform`.

---

## Framework Interop

```python
from medaugmentx.interop import SampleTransform, TorchTransform, MonaiMapTransform
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
| `frameworks` | `torch`, `monai` |
