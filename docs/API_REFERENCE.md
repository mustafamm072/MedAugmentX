# Public surface & semantics

What is covered by compatibility guarantees, and the behaviour that spans more
than one class and therefore has no single docstring to live in.

**Signatures, parameters, and defaults are not on this page.** They are
generated from the source, so they always match the version you have installed:

- [Transform catalog](sphinx/catalog.rst) — every augmentation, what it
  simulates, its knobs, and whether it moves annotations.
- [API reference](sphinx/api/index.rst) — full docstrings for every public
  class and function.

---

## What is public

Prefer these import paths in applications, papers, tutorials, and commercial
code. Internal module paths may move before `1.0` without a deprecation cycle.

```python
from medaugmentx import (                # core data model and pipelines
    MedVolume, Transform, Compose, OneOf, SomeOf,
    PipelineStep, iter_pipeline, pipeline_summary,
    VolumeValidator, Guard, ValidationReport, ValidationIssue, ValidationError,
)
from medaugmentx.transforms import ...   # all augmentations, flat namespace
from medaugmentx.presets import ...      # mri_pipeline, ct_pipeline, dxr_pipeline, dbt_pipeline
from medaugmentx.serialization import ...# REGISTRY, register_transform, to_json, from_json, to_yaml, from_yaml
from medaugmentx.interop import ...      # SampleTransform, TorchTransform, MonaiMapTransform, TorchIOTransform
from medaugmentx.io import ...           # load_dicom_series, load_nifti, save_nifti
from medaugmentx.phantoms import ...     # mri_phantom, ct_phantom, xray_phantom, dbt_phantom
```

One exception is `medaugmentx.core.geometry`, which is public for authors of
custom spatial transforms — see [Geometric targets](#geometric-targets).

`MedVolume`'s `(image, mask, spacing, metadata)` positional signature is
unchanged from earlier releases. The target fields were appended after it, so
existing positional callers keep working.

---

## Geometric targets

`MedVolume` can carry `keypoints` and `bboxes` alongside the image and mask.
This is the one piece of behaviour every spatial transform shares, so it is
specified here rather than repeated in each docstring.

**Layout.** Coordinates are stored in **array-index order** and float dtype —
`(z, y, x)` for 3D and `(y, x)` for 2D, the same order used to index `image`.
Boxes are `(M, 2*ndim)`, laid out low corner first: `[min_axis0, …, max_axis0, …]`.

**Mapping.** Spatial transforms warp targets in lockstep with the pixels:

| Transform | Effect on targets |
| --- | --- |
| `RandomFlip` | Reflect coordinates along the flipped axes |
| `RandomAffine` | Forward affine map about the volume centre, plus translation |
| `ElasticDeform`, `AnisotropicElastic` | Iteratively invert the pixel pull field at each point; raise if inversion does not converge |
| `AnatomicCrop`, `CenterCrop` | Shift by the crop origin (may go negative) |
| `Pad` | Shift by the pad width |
| `Resize` | Align voxel centres using `(new_size - 1) / (old_size - 1)`; singleton axes map to zero |
| `SlabShift` | Translate along the slice axis |
| `CompressionVariation` | Scale about the volume centre on the compression axis |
| `CoarseDropout`, all intensity and artifact transforms | Unchanged (pass through) |

The catalog's *Targets* column is generated from the code. Delegating wrappers
also map targets, even when their own module does not import geometry helpers.

**Boxes under rotation.** A box is mapped through its `2**ndim` corners and
re-bounded to an axis-aligned box, so it stays valid but may grow. For
non-linear displacement fields this is an approximation: the result is not
guaranteed to enclose every warped interior point.

**Targets that leave the frame.** Transforms map targets faithfully and never
drop them, because only the caller knows whether a partly-visible object should
survive. After a crop, call
`volume.remove_out_of_bounds_targets(min_visibility=0.0)` to drop off-frame
keypoints and clip or discard boxes, keeping labels aligned.

**Writing your own.** `medaugmentx.core.geometry` exposes the point maps
(`flip_map`, `affine_map`, `translate_map`, `scale_map`, `displacement_map`) and
the mappers (`map_keypoints`, `map_bboxes`). Build a point map, hand it to
`MedVolume.warp` with the resampled image, and targets are carried for you.

---

## Randomness and reproducibility

Every transform accepts `seed`, and every container derives an **independent**
stream per child rather than sharing one — so two identically-seeded transforms
in the same `Compose` do not draw identical numbers.

A serialised integer seed restarts the *original* draw sequence; it does not
capture the RNG state at the moment of saving. Reloading a pipeline mid-run
therefore replays that stream from the beginning rather than continuing it. For
multi-worker data loaders and full replay, see the
[research guide](RESEARCH_GUIDE.md).

---

## Validation severity

`VolumeValidator` rules fall into two groups:

- **Absolute** rules look at the augmented volume alone: finiteness, collapse to
  a constant, mask shape, intensity bounds, minimum foreground fraction.
- **Comparative** rules need the pre-augmentation volume — label preservation,
  foreground loss, intensity drift. They are skipped unless you pass
  `reference=` to `validate()`. `Guard` supplies it automatically.

A `ValidationReport` is truthy when the volume passed, meaning it carries no
error-severity issues. Bounds violations are warnings by default and become
errors under `strict_bounds=True`. In `on_fail="raise"` mode `Guard` raises
`ValidationError`, which carries the failing report on `.report`.

---

## Serialisation limits

Pipelines round-trip through JSON and YAML so the exact augmentation policy can
be committed next to a set of results. Loading enforces hard limits: at most
1,000,000 characters of text, 64 levels of nesting, and 10,000 transforms.
Malformed structures are rejected rather than partially constructed.

Only classes present in `REGISTRY` are ever instantiated. Registering a custom
transform makes its constructor reachable by name from any document you load —
custom registrations are trusted Python code, so treat pipeline files from
outside your project as executable input. See the
[security policy](../SECURITY.md).

---

## Framework interop

The adapters accept the sample shapes each framework uses and return the same
kind they were given:

| Input | Output |
| --- | --- |
| `MedVolume` | `MedVolume` |
| `np.ndarray` or `torch.Tensor` image | Same array or tensor kind |
| `(image, mask)` tuple or list | Same tuple or list kind |
| Mapping with `image_key` and optional `mask_key` | Shallow-copied mapping |
| TorchIO `Subject` | `Subject`, with augmented data written back |

MedAugmentX operates on single-channel 2D and 3D data. `channel_dim` — `None`,
`"auto"`, `"first"`, `"last"`, or an integer axis — says where the adapter
should expect a singleton channel dimension; it is stripped before augmentation
and restored afterwards.

For a TorchIO `Subject`, explicit `image_key` and `label_key` are used when
present; otherwise the adapter infers a unique non-label image key and a unique
label-like key. Pass explicit keys for multi-image subjects.

Importing an adapter never imports its framework. Torch, MONAI, and TorchIO are
touched only when a sample of that kind actually passes through.

---

## Optional extras

The core install is NumPy and SciPy only. Everything else is opt-in:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `dicom` | `pydicom` | `load_dicom_series` |
| `nifti` | `nibabel` | `load_nifti`, `save_nifti` |
| `io` | `pydicom`, `nibabel` | Both readers |
| `yaml` | `pyyaml` | `to_yaml`, `from_yaml` |
| `torch` | `torch` | `TorchTransform` with tensors |
| `monai` | `monai` | MONAI dictionary pipelines |
| `torchio` | `torchio` | `TorchIOTransform` |
| `frameworks` | `torch`, `monai`, `torchio` | All three |
| `notebooks` | `matplotlib`, `nbformat`, `nbclient`, `ipykernel` | Running the tutorials |
| `docs` | `sphinx`, `myst-parser`, `furo` | Building this site |

---

## Related

| Document | Use |
| --- | --- |
| [Commercial adoption](COMMERCIAL_ADOPTION.md) | Intended use, validation, audit trail, privacy, dependency guidance |
| [Architecture](ARCHITECTURE.md) | Layering, contracts, dependency boundaries, testing strategy |
| [Research guide](RESEARCH_GUIDE.md) | Replay, worker seeds, evaluation reporting, citation |
| [Security policy](../SECURITY.md) | Vulnerability reporting, PHI handling, dependency posture |
