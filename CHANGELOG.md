# Changelog

All notable changes to MedAugmentX will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.6.0] — 2026-06-19

### Added

This release roughly doubles the transform library — from 22 to 36 registered
transforms — completing the deferred Phase 3 modality artifacts and adding a
set of general-purpose transforms for parity with (and breadth beyond)
comparable 3D medical augmentation libraries. Every new transform is seedable,
serialisable, mask-safe, and covered by unit tests.

**Spatial transforms** (`medaugmentx.transforms.spatial`)
- `CoarseDropout` — cutout-style random rectangular/box occlusion (2D/3D),
  optional mask blanking.
- `Resize` — resample to a fixed shape; mask uses nearest-neighbour and
  `spacing` is rescaled to match the new voxel grid.
- `Pad` — centre-pad up to a target shape (never crops).
- `CenterCrop` — centre-crop to a target shape (never pads). Pair with `Pad`
  to force an exact shape for batching.

**Intensity transforms** (`medaugmentx.transforms.intensity`)
- `MedianBlur` — edge-preserving median filter (salt-and-pepper / speckle).
- `Sharpen` — unsharp-mask edge enhancement.
- `CLAHEContrast` — Contrast Limited Adaptive Histogram Equalization with
  bilinear tile interpolation (pure NumPy, applied per-slice for 3D).
- `HistogramMatch` — match the intensity histogram to a reference distribution,
  with a `blend` ratio; reference serialises inline (or `None` for identity).

**Modality transforms — MRI** (`medaugmentx.transforms.modality.mri`)
- `MRIMotion` — in-plane rigid-body motion blur/ghosting (averaged motion
  states).

**Modality transforms — CT** (`medaugmentx.transforms.modality.ct`)
- `MetalStreak` — radiating bright/dark streak artifact from dense implants.

**Modality transforms — X-ray** (`medaugmentx.transforms.modality.xray`, new)
- `ScatterSimulation` — low-frequency scatter (veiling glare) that lowers
  contrast.
- `GridArtifact` — stationary anti-scatter grid line pattern.

**Modality transforms — Tomosynthesis** (`medaugmentx.transforms.modality.tomosynthesis`)
- `CompressionVariation` — anisotropic breast-paddle compression variation
  (mask-consistent in-plane scaling).
- `ReconStreak` — limited-angle out-of-plane reconstruction streaks
  (parallax replicas across neighbouring planes).

### Changed

- Preset pipelines now incorporate the new artifacts: `mri_pipeline` adds
  `MRIMotion` to its artifact `OneOf`; `ct_pipeline` adds occasional
  `MetalStreak`; `dxr_pipeline` adds `CLAHEContrast` and a scatter/grid
  `OneOf`; `dbt_pipeline` adds `CompressionVariation` and `ReconStreak`.
- All 14 new transforms are registered in `serialization.REGISTRY` and
  re-exported from `medaugmentx.transforms`.
- Version bumped to `0.6.0`.

### Documentation

- README, API reference, architecture, and milestones updated for the
  expanded transform library and the new X-ray modality module.
- Roadmap items 3.8 (remaining deferred transforms) and 3.9 (benchmark suite)
  marked complete.

### Tooling

- Added `benchmarks/benchmark.py`, a dependency-free per-transform speed
  benchmark with a configurable volume shape, plus `benchmarks/README.md`
  documenting the CPU 500 ms target and the still-planned GPU backend.

## [0.5.0] — 2026-06-15

### Added

- `medaugmentx.serialization.register_transform` — a decorator for registering
  custom transforms for JSON/YAML round-trips. Validates that the class is a
  `Transform` subclass and refuses to silently overwrite an existing registry
  entry (e.g. a built-in) unless `override=True` is passed. Usable bare
  (`@register_transform`) or parametrised (`@register_transform(name=...,
  override=...)`). Direct assignment to `REGISTRY` continues to work for
  backward compatibility.

### Changed

- Documentation and the `custom_transform` example now recommend
  `@register_transform` over manual `REGISTRY` mutation for custom transforms.

## [0.4.0] — Phase 3 TorchIO Interop

### Added

**Framework interop** (`medaugmentx.interop`)
- `TorchIOTransform` — optional TorchIO `Subject` adapter for one intensity
  image plus one optional label map.
- Key inference for simple TorchIO subjects and explicit `image_key` /
  `label_key` controls for multi-image studies.
- Subject/image copy handling so the default adapter path returns augmented
  TorchIO-like objects without mutating the caller's original object when
  those objects provide `copy()`.

**Packaging**
- Version bumped to `0.4.0`.
- New optional extra: `[torchio]`.
- `[frameworks]` now installs PyTorch, MONAI, and TorchIO integrations.

**Documentation**
- Updated README, API reference, API examples, architecture, and milestones
  for TorchIO interop.
- Added a commercial adoption guide covering intended use, dependency policy,
  reproducibility, audit trails, validation, and privacy expectations.
- Added `SECURITY.md` for vulnerability reporting, PHI handling, dependency
  posture, and clinical safety boundaries.
- Added a docs index and tightened API example wording for accuracy.
- Added source-distribution manifest entries and package metadata links for
  adoption and security documentation.

## [0.3.0] — Phase 3 Developer Interop

### Added

**Framework interop** (`medaugmentx.interop`)
- `SampleTransform` — adapts any MedAugmentX transform or pipeline to
  `MedVolume`, image arrays/tensors, `(image, mask)` tuples/lists, and
  mapping samples.
- `TorchTransform` — PyTorch / torchvision-friendly alias that supports torch
  tensors at runtime without importing torch during package import.
- `MonaiMapTransform` — MONAI-style dict adapter with `image` / `label`
  defaults.
- Singleton channel handling via `channel_dim`, with automatic restoration
  after augmentation.
- Mask/label dtype preservation by default.

**Packaging**
- Version bumped to `0.3.0`.
- Package now includes `py.typed` for PEP 561 type-checker discovery.
- New optional extras: `[torch]`, `[monai]`, and `[frameworks]`.
- Added healthcare and typed-package PyPI classifiers.

**Documentation**
- Added `docs/API_REFERENCE.md` for developer-facing public API docs.
- Updated README, API examples, architecture, milestones, and examples docs
  for `0.3.0`.
- Fixed stale `medaugment` import examples to use `medaugmentx`.

## [0.2.0] — Phase 2

### Added

**Intensity transforms**
- `BiasField` — smooth multiplicative MRI bias field (RF coil / B0 inhomogeneity).
- `WindowLevel` — random window/level perturbation for CT protocol variation.
- `BrightnessContrast` — additive brightness + multiplicative contrast, native intensity space.
- `GaussianBlur` — isotropic Gaussian blur with sigma range.
- `SimulateLowResolution` — downsample + upsample to simulate cross-site resolution variation.

**Modality transforms — MRI** (`medaugmentx.transforms.modality.mri`)
- `GhostingArtifact` — phase-encoding ghosting (shifted attenuated replica).
- `KSpaceDropout` — random k-space line zeroing with correct Gibbs ringing reconstruction.

**Modality transforms — CT** (`medaugmentx.transforms.modality.ct`)
- `BeamHardening` — radially-symmetric cupping artifact simulation.

**Serialisation** (`medaugmentx.serialization`)
- `to_json` / `from_json` — lossless JSON round-trip for any transform or pipeline.
- `to_yaml` / `from_yaml` — optional YAML round-trip (requires `pip install pyyaml`).
- `REGISTRY` — dict mapping class names to classes; extend for custom transforms.
- All built-in transforms override `to_dict()` to produce round-trippable dicts.
- `Compose`, `OneOf`, `SomeOf` now serialise children recursively.

**Presets** (`medaugmentx.presets`)
- `mri_pipeline(seed)` — MRI spatial + bias field + Rician noise + optional ghosting.
- `ct_pipeline(seed)` — CT spatial + window/level + Gaussian noise + beam hardening.
- `dxr_pipeline(seed)` — Digital X-ray spatial + blur + brightness/contrast + gamma.
- `dbt_pipeline(seed)` — DBT full pipeline combining Phase-1 DBT transforms with bias field.

**Packaging**
- Version bumped to `0.2.0`.
- New optional extra `[yaml]` for PyYAML support.

## [0.1.0] — Phase 1 MVP

### Added
- Core data model: `MedVolume` dataclass, `Transform` ABC, RNG helpers.
- Pipeline primitives: `Compose`, `OneOf`, `SomeOf` with end-to-end
  deterministic seeding.
- Spatial transforms: `RandomAffine`, `RandomFlip`, `AnatomicCrop`,
  `ElasticDeform` (anisotropic sigma).
- Intensity transforms: `RicianNoise`, `GaussianNoise`, `GammaCorrection`.
- Tomosynthesis (DBT) Phase 1 transforms: `SlabShift`, `LimitedAngleBlur`,
  `SliceDropout`, `AnisotropicElastic`.
- I/O: DICOM series loader (`load_dicom_series`) and NIfTI reader/writer
  (`load_nifti`, `save_nifti`).
- pytest test suite, type hints, GitHub Actions CI.
