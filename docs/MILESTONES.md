# Milestones & Roadmap

MedAugmentX ships in three phases. Phases 1 and 2 are complete; Phase 3 is
in progress and focuses on wider developer adoption, optional framework
interop, GPU acceleration, comprehensive documentation, and the v1.0 release.

---

## Phase 1 — Core MVP ✅

**Goal:** A pip-installable library with the most essential transforms and
DICOM/NIfTI I/O. Enough for early adopters and internal testing.

### Deliverables

| # | Deliverable | Module | Status |
| --- | --- | --- | --- |
| 1.1 | `MedVolume` dataclass + `Transform` ABC | `medaugmentx/core/{volume,base}.py` | ✅ |
| 1.2 | `Compose`, `OneOf`, `SomeOf` | `medaugmentx/core/compose.py` | ✅ |
| 1.3 | `RandomAffine`, `RandomFlip`, `AnatomicCrop` | `medaugmentx/transforms/spatial/` | ✅ |
| 1.4 | `ElasticDeform` with anisotropic sigma | `medaugmentx/transforms/spatial/elastic.py` | ✅ |
| 1.5 | `RicianNoise`, `GaussianNoise`, `GammaCorrection` | `medaugmentx/transforms/intensity/` | ✅ |
| 1.6 | DICOM loader with spacing metadata | `medaugmentx/io/dicom.py` | ✅ |
| 1.7 | NIfTI loader/writer | `medaugmentx/io/nifti.py` | ✅ |
| 1.8 | `SlabShift` + `LimitedAngleBlur` (DBT) | `medaugmentx/transforms/modality/tomosynthesis/` | ✅ |
| 1.9 | `SliceDropout`, `AnisotropicElastic` (DBT) | `medaugmentx/transforms/modality/tomosynthesis/` | ✅ |
| 1.10 | Unit + integration test suite (pytest, 92 tests) | `tests/` | ✅ |
| 1.11 | PyPI packaging + README docs | `pyproject.toml`, `README.md` | ✅ |

---

## Phase 2 — Modality Artifacts & Serialisation ✅

**Goal:** Physics-based artifact simulation for all major modalities, a full
serialisation layer, and pre-built preset pipelines.

### Deliverables

| # | Deliverable | Module | Status |
| --- | --- | --- | --- |
| 2.1 | `BiasField` (smooth multiplicative MRI field) | `transforms/intensity/bias_field.py` | ✅ |
| 2.2 | `WindowLevel` (CT protocol variation) | `transforms/intensity/window_level.py` | ✅ |
| 2.3 | `BrightnessContrast`, `GaussianBlur`, `SimulateLowResolution` | `transforms/intensity/blur.py`, `brightness_contrast.py` | ✅ |
| 2.4 | `GhostingArtifact`, `KSpaceDropout` (MRI) | `transforms/modality/mri/` | ✅ |
| 2.5 | `BeamHardening` (CT cupping artifact) | `transforms/modality/ct/` | ✅ |
| 2.6 | `to_dict()` round-trip overrides on all transforms | all transform modules | ✅ |
| 2.7 | `Compose`/`OneOf`/`SomeOf` recursive `to_dict()` | `medaugmentx/core/compose.py` | ✅ |
| 2.8 | JSON serialisation: `to_json`, `from_json`, `REGISTRY` | `medaugmentx/serialization.py` | ✅ |
| 2.9 | Optional YAML serialisation: `to_yaml`, `from_yaml` | `medaugmentx/serialization.py` | ✅ |
| 2.10 | Preset pipelines: `mri_pipeline`, `ct_pipeline`, `dxr_pipeline`, `dbt_pipeline` | `medaugmentx/presets.py` | ✅ |
| 2.11 | Extended test suite (216 tests total) | `tests/` | ✅ |

### Remaining Phase 2 work (deferred to Phase 3)

The following items from the original Phase 2 scope were intentionally
deferred to keep the release focused and the library lightweight. The
augmentation transforms shipped in `0.6.0`:

- Vendor-specific DICOM parsers (Hologic, GE, Siemens multi-frame DBT) — *still planned (3.7)*
- `MRIMotion` (in-plane rigid motion blur) — ✅ `0.6.0`
- `MetalStreak` (CT metal artifact simulation) — ✅ `0.6.0`
- `ScatterSimulation`, `GridArtifact` (X-ray) — ✅ `0.6.0`
- `CompressionVariation`, `ReconStreak` (DBT reconstruction) — ✅ `0.6.0`
- `CLAHEContrast`, `HistogramMatch` (advanced intensity) — ✅ `0.6.0`
- Anatomical plausibility validator — ✅ `0.8.0` (`VolumeValidator` + `Guard`)

---

## Phase 3 — Framework Interop, GPU & v1.0

**Goal:** Make the library easier to adopt in academic and commercial
training stacks while keeping the default install lightweight.

### Deliverables

| # | Deliverable | Module | Status |
| --- | --- | --- | --- |
| 3.1 | Lightweight framework adapters for PyTorch/torchvision and MONAI-style samples | `medaugmentx/interop/` | ✅ `0.3.0` |
| 3.2 | Typed package marker for downstream type checkers | `medaugmentx/py.typed` | ✅ `0.3.0` |
| 3.3 | Developer API reference and updated examples | `docs/API_REFERENCE.md`, `docs/API_EXAMPLES.md` | ✅ `0.3.0` |
| 3.4 | PyTorch backend for GPU-accelerated spatial transforms | `backends/torch/` | Planned |
| 3.5 | TorchIO subject adapter and MONAI-style dictionary integration | `interop/` | Adapter surface shipped `0.4.0`; real patient-space integration validation pending |
| 3.6 | Commercial adoption and security guidance | `docs/COMMERCIAL_ADOPTION.md`, `SECURITY.md` | ✅ `0.4.0` |
| 3.7 | Vendor DICOM parsers — Hologic, GE, Siemens | `io/dicom_vendor/` | Planned |
| 3.8 | Remaining deferred transforms (motion, metal, scatter, grid, CLAHE, histogram match, compression, recon streak) + general transforms (cutout, resize/pad/crop, median blur, sharpen) | `transforms/` | ✅ `0.6.0` |
| 3.9 | Benchmark suite (per-transform speed) | `benchmarks/` | ✅ `0.6.0` |
| 3.10 | Pipeline inspection for experiment logs and augmentation policy review | `medaugmentx/inspection.py` | ✅ `0.7.0` |
| 3.11 | Plausibility validation + safe-augmentation guard (`VolumeValidator`, `Guard`) | `medaugmentx/validation.py` | ✅ `0.8.0` |
| 3.12 | Keypoint & bounding-box targets warped through all spatial transforms | `medaugmentx/core/geometry.py`, `core/volume.py` | ✅ `0.9.0` |
| 3.13 | Sphinx documentation site | `docs/sphinx/`, `.github/workflows/docs.yml` | ✅ `0.10.0` build; published to GitHub Pages (unreleased) |
| 3.14 | Jupyter tutorials (MRI, CT, DBT) | `notebooks/`, `medaugmentx/phantoms.py` | ✅ unreleased |
| 3.15 | v1.0 release + GitHub Actions CD pipeline | `.github/` | Prepared for 1.0.0; publication requires maintainer setup |
| 3.16 | Coherent patient-space affine, units, resampling, and DICOM geometry contracts | `core/`, `io/`, spatial transforms | Implemented in 1.0.0; see release evidence |
| 3.17 | Actual framework-object integration CI and quantitative modality-model checks | `tests/`, `.github/` | Implemented in 1.0.0; see release evidence |

### Acceptance criteria (Phase 3 / v1.0)

- All Phase 1 and 2 deliverables shipped.
- Default installation keeps only NumPy and SciPy as hard dependencies.
- Framework-specific integrations are optional and import-lazy.
- Commercial adoption docs cover intended use, dependency policy, audit
  trails, privacy/security posture, and clinical validation expectations.
- Patient-space coordinates, affine, spacing, and units agree through I/O,
  crop/pad/resize, and export, including oblique and anisotropic volumes.
- Unsupported enhanced/multiframe DICOM objects fail explicitly until their
  per-frame geometry and scaling are supported.
- Optional integration jobs validate real framework objects and their geometry.
- Preset intensity domains, voxel/mm units, laterality rules, and model
  assumptions are documented and supported by quantitative regression tests.
- CPU benchmarks record measured latency/memory without universal timing claims.
- Release workflow verifies tag/version and publishes only on a published release.
- GPU acceleration is future work outside the v1 CPU API.

---

## Issue labels

Issues and pull requests use the labels below. Phases map 1:1 to GitHub
milestones (`phase-1`, `phase-2`, `phase-3`).

| Label | Use |
| --- | --- |
| `transform` | New augmentation primitive |
| `io` | DICOM / NIfTI / new format |
| `dbt` | Tomosynthesis-specific |
| `bug` | Defect |
| `docs` | Documentation only |
| `phase-3` | Roadmap milestone |

Contributors: see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Current adoption priorities

Version 1 implements the geometry, integration, policy, numerical-validation,
QA and release-tooling work from the [October review](PROJECT_REVIEW.md).
See [release readiness](RELEASE_READINESS.md) for measured evidence and remaining
external checks. Real patient/site evaluation and vendor DICOM fixtures remain
needed before task-specific clinical or scanner-compatibility claims. GPU
acceleration remains optional future work, not a v1 promise.
