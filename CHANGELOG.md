# Changelog

All notable changes to MedAugment will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] — Phase 2

### Added

**Intensity transforms**
- `BiasField` — smooth multiplicative MRI bias field (RF coil / B0 inhomogeneity).
- `WindowLevel` — random window/level perturbation for CT protocol variation.
- `BrightnessContrast` — additive brightness + multiplicative contrast, native intensity space.
- `GaussianBlur` — isotropic Gaussian blur with sigma range.
- `SimulateLowResolution` — downsample + upsample to simulate cross-site resolution variation.

**Modality transforms — MRI** (`medaugment.transforms.modality.mri`)
- `GhostingArtifact` — phase-encoding ghosting (shifted attenuated replica).
- `KSpaceDropout` — random k-space line zeroing with correct Gibbs ringing reconstruction.

**Modality transforms — CT** (`medaugment.transforms.modality.ct`)
- `BeamHardening` — radially-symmetric cupping artifact simulation.

**Serialisation** (`medaugment.serialization`)
- `to_json` / `from_json` — lossless JSON round-trip for any transform or pipeline.
- `to_yaml` / `from_yaml` — optional YAML round-trip (requires `pip install pyyaml`).
- `REGISTRY` — dict mapping class names to classes; extend for custom transforms.
- All built-in transforms override `to_dict()` to produce round-trippable dicts.
- `Compose`, `OneOf`, `SomeOf` now serialise children recursively.

**Presets** (`medaugment.presets`)
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
