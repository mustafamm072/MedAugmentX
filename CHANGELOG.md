# Changelog

All notable changes to MedAugmentX will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — prepared 2026-10-25

### Changed

- Establish array-index-to-RAS-mm affine geometry, coherent crop/pad/resize and
  NIfTI image/mask export with spatial-unit conversion and shear preservation.
- Resize now uses voxel-extent coordinates and anti-aliasing. Affine rotation
  defaults to the physical metric. See `docs/MIGRATION_V1.md` for breaking changes.
- Strict classic single-frame DICOM validation, regular shear support, explicit
  skip diagnostics, and opt-in PatientID retention.
- Presets disable flips by default; optional normalization, CT HU fill and
  in-plane DBT scale constraints. Explicit involutive laterality remapping.
- Elastic boundary targets use continuous field extension independently of
  constant image fill; conservative boxes bypass unnecessary corner inversion.
- Bounded elastic gradients, optional Jacobian rejection, conservative boxes,
  finite/range/size validation and per-class physical foreground checks.

### Added

- Schema/version/hash policy envelopes, stable per-sample seeding, exact RNG
  checkpoint replay, pipeline ownership checks and Guard rejection telemetry.
- Optional local preview/QA, percentile normalization, packaged model cards for
  every transform, real framework and spawned-worker integration tests.
- Measured JSON latency/memory benchmarks and a synthetic segmentation ablation
  reporting a negative appearance-augmentation result; no clinical benefit claim.
- Minimum-dependency, OS, framework, branch coverage and format CI jobs; offline
  docs builds and an explicit release-event trusted-publishing workflow.
- Single version source and v1 migration/release evidence guides.

The following documentation resources were developed during pre-release work:


### Added

- Generated API reference. `sphinx.ext.autodoc` + `napoleon` now render every
  public class and function from its own docstring, across five pages: core data
  model, transforms, presets/serialisation/inspection, validation, and
  I/O/interop/phantoms. Signatures, parameter types, and defaults come from the
  source, so they describe the installed version rather than whatever was last
  transcribed by hand.
- Generated transform catalog at `docs/sphinx/catalog.rst` — a scannable table
  per modality giving each transform's summary, parameters, and whether it warps
  keypoints and boxes. Built by the new `docs/sphinx/_ext/mx_catalog.py`
  extension, which reads `medaugmentx.transforms.__all__`, the class docstrings,
  the constructor signatures, and (for the targets column) whether the defining
  module uses `medaugmentx.core.geometry`. Adding a transform is enough to make
  it appear.
- `tests/test_docs.py` guards what the generated pages read: every exported
  transform must fall into a catalog category, carry a summary sentence, and
  document each of its constructor parameters. The targets column is checked
  against the behaviour specified in `docs/API_REFERENCE.md`.
- Furo theme (`docs` extra) supplying grouped sidebar navigation, full-text
  search, light/dark switching, per-page contents, and working "Edit this page"
  links.
- Jupyter tutorials for MRI, CT, and tomosynthesis in `notebooks/`, covering the
  modality-specific reasoning behind each transform plus mask alignment,
  seeding, guarding, and policy serialisation (roadmap 3.14).
- `medaugmentx.phantoms` — synthetic MRI, CT, X-ray, and DBT volumes generated
  from analytic shapes with no downloads or patient data. Promoted out of the
  README gallery script so tutorials, examples, and tests share one definition.
- `notebooks` optional extra (`matplotlib`, `nbformat`, `nbclient`, `ipykernel`)
  and `notebooks/run_notebooks.py`, which CI runs on every push so the tutorials
  cannot rot.
- `.github/workflows/docs.yml` publishes the Sphinx site to GitHub Pages on
  push to `main`, completing roadmap 3.13. The site is live at
  <https://mustafamm072.github.io/MedAugmentX/>, and the project's
  `Documentation` URL and README badge now point at it.

### Fixed

- CT presets and recipes keep windowed data in its native HU intensity scale
  before adding HU-scale noise. Model-input normalization is a separate step.
- `RandomAffine` rotation labels now follow `(z, y, x)` array order; DBT
  presets rotate about `z` for in-plane rotation. DBT elastic defaults and
  teaching examples put the smallest displacement on the slice axis.
- Elastic displacement components all use the spatial sigma tuple. Landmark
  mapping iteratively inverts the pixel pull field and raises on nonconvergence
  instead of returning the one-step `p - d(p)` approximation.
- `MedVolume` rejects empty spatial axes, invalid spacing, and nonfinite
  geometric targets. Scalar and range gamma values share finite/positive
  validation. Low-resolution simulation retains non-empty temporary axes.
- Pipeline inspection traverses guarded children. The transform catalog
  recognizes target mapping delegated by `AnisotropicElastic`, and custom
  transform examples retain integer constructor seeds.
- The documentation site root no longer 404s. Sphinx writes its root document
  to `docs/sphinx/index.html` because the source directory is the repository
  root; `docs/sphinx/_extra/` now supplies a redirect (and `.nojekyll`) at the
  output root, and CI fails the docs build if it goes missing.

### Changed

- Corrected geometry and preset behavior changes seeded outputs. Reproduce
  previous experiments with their original package/environment; serialized
  configuration does not guarantee identical behavior across library versions.
- Added `docs/PROJECT_REVIEW.md` with reproduced defects, remaining release
  blockers, and prioritized acceptance criteria. Documentation distinguishes
  voxel-space settings and artifact approximations from validated physical
  simulation and patient-space guarantees.
- The documentation site is organised by task rather than presented as a flat
  list of files. The landing page orients the reader and routes them from a
  goal to a page; the sidebar groups everything under Getting started,
  Reference, Guides, and Project.
- `docs/API_REFERENCE.md` is now *Public surface & semantics*. Its
  constructor-signature tables — which duplicated the code and had to be updated
  by hand on every change — were removed in favour of the generated pages, and
  what remains is the behaviour that spans several classes: geometric target
  layout and mapping, seeding guarantees, validation severity, serialisation
  limits, interop sample shapes, and the extras table.
- `docs/README.md` is now a guide to working on the documentation — how the
  site is assembled, what is generated versus hand-written, and how to add a
  page — instead of a second copy of the site's navigation.
- Releases no longer require editing a version line in `docs/API_REFERENCE.md`;
  the site reads its version from the installed package. `CONTRIBUTING.md`'s
  release checklist drops from six files to five.
- `examples/generate_readme_gallery.py` imports its phantoms from
  `medaugmentx.phantoms` instead of defining them inline. Gallery output is
  byte-identical.

## [0.10.0] — 2026-09-05

### Fixed

- Keep the copied test tree named `tests` in the isolated-wheel CI job; `medaugmentx-tests` is not an importable package name, so pytest could not collect it.
- Write DICOM test fixtures with explicit little-endian encoding on both pydicom 2.x and 3.x. pydicom 2.x (the newest release available on Python 3.9) does not infer the dataset encoding from `file_meta`.
- Preserve integer seeds in every built-in transform configuration, including standalone JSON/YAML round-trips.
- Align resize landmarks/boxes with the sampled voxel-centre grid; move targets through DBT slab shifts and compression.
- Isolate each Guard attempt from input mutation, deep-copy nested metadata, and detect removed masks. Clarify that retries counts total attempts.
- Return updated spacing/metadata from dictionary adapters.
- Use physical DICOM spacing rather than instance-number differences; reject duplicate/irregular positions and color images, and apply per-slice rescale defaults.
- Reject malformed/deep/oversized pipeline configurations and invalid registry entries; reject non-finite JSON constants and invalid selection weights.

### Added

- Release checklist in `CONTRIBUTING.md` naming every file that records the version number.
- README before/after gallery for MRI, CT, X-ray, and DBT, generated from synthetic phantoms with a reproducible script and parameter manifest.
- Searchable Sphinx documentation from the existing Markdown guides, with CI preview artifacts.
- Research reproducibility guide, software citation metadata, and a runnable synthetic experiment record.
- Distribution build/metadata checks and tests against an installed wheel outside the checkout.
- CI execution of adoption examples and 2D/3D benchmark smoke checks; repaired benchmark shape handling and stale Guard registration handling.

### Documentation

- Restore the changelog preamble to the top of the file and merge the two `Unreleased` headings into one.
- Record `version`, `date-released`, and the archive DOI in `CITATION.cff` so the generated citation matches the README badge.
- Correct the clone URL in `CONTRIBUTING.md`.

### Compatibility notes

Saved policies now include leaf seeds. Loading a saved integer-seeded standalone transform starts its sequence again; it does not restore current RNG state. Resize target coordinates change to match existing image interpolation. Guard now copies inputs for isolation, increasing peak memory. DICOM series with irregular or duplicate physical slice positions now fail explicitly. Text policies above 1,000,000 characters require reduction or trusted in-process construction. No release or public deployment is performed by these changes.

## [0.9.0] — 2026-07-23

### Added

- Keypoint and bounding-box targets on `MedVolume`. Every volume can now carry
  `keypoints` (`(N, ndim)` landmark coordinates), `bboxes` (`(M, 2*ndim)`
  axis-aligned boxes laid out as `[min…, max…]`), and optional parallel
  `keypoint_labels` / `bbox_labels`. Coordinates use array-index order
  (`(z, y, x)` for 3D, `(y, x)` for 2D) so they line up with the image axes.
- Geometric targets are warped in lockstep with the image by every spatial
  transform — `RandomFlip`, `RandomAffine`, `ElasticDeform`, `AnatomicCrop`,
  `Resize`, `Pad`, and `CenterCrop`. Boxes are transformed via their corners
  and re-bounded to a valid axis-aligned box, so they stay correct under
  rotation. `CoarseDropout` and every intensity/artifact transform pass targets
  through untouched.
- `medaugmentx.core.geometry` — a dependency-free module of coordinate helpers
  (`flip_map`, `affine_map`, `translate_map`, `scale_map`, `displacement_map`,
  `map_keypoints`, `map_bboxes`) that custom spatial transforms can reuse.
- `MedVolume.warp(point_map, *, image, mask=None, spacing=None)` — the single
  entry point spatial transforms use to swap in a warped image while mapping
  targets; `MedVolume.remove_out_of_bounds_targets(min_visibility=0.0)` prunes
  keypoints that left the frame and clips/drops boxes after a crop, keeping
  labels aligned.
- `MedVolume` gains `has_keypoints`, `has_bboxes`, `num_keypoints`, and
  `num_bboxes` conveniences; `repr` now reports target counts.
- `examples/keypoints_bboxes.py` demonstrating landmark and box tracking
  through a spatial pipeline.

### Changed

- Version bumped to `0.9.0`.
- `MedVolume.replace()` and `MedVolume.copy()` now carry the four new target
  fields; `None` still means "keep current", so intensity transforms preserve
  targets automatically.

### Notes

- Modality-specific artifact transforms that resample voxels (`SlabShift`,
  `AnisotropicElastic`) pass targets through unchanged for now — thread them
  through `MedVolume.warp` if you need target tracking there.
- Serialisation is unaffected: targets live on the volume, not on transforms,
  so the JSON/YAML `REGISTRY` and round-trips are unchanged.

## [0.8.0] — 2026-07-09

### Added

- `medaugmentx.VolumeValidator` — a clinically-aware plausibility validator
  that checks an augmented `MedVolume` for non-finite values, collapse to a
  constant, mask/image shape desync, out-of-range intensity, foreground
  fraction/loss, lost mask labels, and distribution drift. Comparative checks
  run against the pre-augmentation volume when a reference is supplied.
- `medaugmentx.Guard` — a drop-in `Transform` wrapper that validates the
  output of any transform or pipeline on every call and, on failure, can
  `raise`, `warn`, `revert` to the untouched input, or `retry` with fresh
  randomness. Guards nest inside `Compose`/`OneOf`/`SomeOf` and round-trip
  through JSON/YAML like every other transform.
- `medaugmentx.ValidationReport`, `medaugmentx.ValidationIssue`, and
  `medaugmentx.ValidationError` for reading and reacting to validation results.
- `examples/safe_augmentation.py` demonstrating direct validation and all four
  guard failure modes.

### Changed

- Version bumped to `0.8.0`.
- `Guard` is registered in the serialisation `REGISTRY`; `from_dict` now
  reconstructs its single wrapped transform and rebuilds its validator.

### Fixed

- README DOI badge now uses the Zenodo DOI-form URL
  (`zenodo.org/badge/DOI/…`) so it renders the concept DOI instead of showing a
  placeholder question mark.

### Documentation

- README, API reference, and the milestones roadmap document the new
  validation and safe-augmentation workflow; the long-deferred "anatomical
  plausibility validator" item is marked shipped.

## [0.7.0] — 2026-06-29

### Added

- `medaugmentx.pipeline_summary` and `medaugmentx.iter_pipeline` for
  human-readable and programmatic inspection of transforms and nested
  pipelines. The helpers are backed by each transform's `to_dict()` structure
  so summaries stay aligned with JSON/YAML serialisation.
- `medaugmentx.PipelineStep`, a small dataclass describing each inspected
  pipeline node (`path`, `name`, `params`, `depth`).

### Changed

- Version bumped to `0.7.0`.

### Documentation

- README, API reference, and API examples now document pipeline inspection for
  experiment logs and augmentation policy review.

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
