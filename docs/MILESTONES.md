# Milestones & Roadmap

MedAugment ships in three phases. Phases 1 and 2 are complete; Phase 3
builds GPU acceleration, framework integrations, and the v1.0 release.

---

## Phase 1 — Core MVP ✅

**Goal:** A pip-installable library with the most essential transforms and
DICOM/NIfTI I/O. Enough for early adopters and internal testing.

### Deliverables

| # | Deliverable | Module | Status |
| --- | --- | --- | --- |
| 1.1 | `MedVolume` dataclass + `Transform` ABC | `medaugment/core/{volume,base}.py` | ✅ |
| 1.2 | `Compose`, `OneOf`, `SomeOf` | `medaugment/core/compose.py` | ✅ |
| 1.3 | `RandomAffine`, `RandomFlip`, `AnatomicCrop` | `medaugment/transforms/spatial/` | ✅ |
| 1.4 | `ElasticDeform` with anisotropic sigma | `medaugment/transforms/spatial/elastic.py` | ✅ |
| 1.5 | `RicianNoise`, `GaussianNoise`, `GammaCorrection` | `medaugment/transforms/intensity/` | ✅ |
| 1.6 | DICOM loader with spacing metadata | `medaugment/io/dicom.py` | ✅ |
| 1.7 | NIfTI loader/writer | `medaugment/io/nifti.py` | ✅ |
| 1.8 | `SlabShift` + `LimitedAngleBlur` (DBT) | `medaugment/transforms/modality/tomosynthesis/` | ✅ |
| 1.9 | `SliceDropout`, `AnisotropicElastic` (DBT) | `medaugment/transforms/modality/tomosynthesis/` | ✅ |
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
| 2.7 | `Compose`/`OneOf`/`SomeOf` recursive `to_dict()` | `medaugment/core/compose.py` | ✅ |
| 2.8 | JSON serialisation: `to_json`, `from_json`, `REGISTRY` | `medaugment/serialization.py` | ✅ |
| 2.9 | Optional YAML serialisation: `to_yaml`, `from_yaml` | `medaugment/serialization.py` | ✅ |
| 2.10 | Preset pipelines: `mri_pipeline`, `ct_pipeline`, `dxr_pipeline`, `dbt_pipeline` | `medaugment/presets.py` | ✅ |
| 2.11 | Extended test suite (216 tests total) | `tests/` | ✅ |

### Remaining Phase 2 work (deferred to Phase 3)

The following items from the original Phase 2 scope were intentionally
deferred to keep the release focused and the library lightweight:

- Vendor-specific DICOM parsers (Hologic, GE, Siemens multi-frame DBT)
- `MRIMotion` (in-plane rigid motion blur)
- `MetalStreak` (CT metal artifact simulation)
- `ScatterSimulation`, `GridArtifact` (X-ray)
- `CompressionVariation`, `ReconStreak` (DBT reconstruction)
- `CLAHEContrast`, `HistogramMatch` (advanced intensity)
- Anatomical plausibility validator

---

## Phase 3 — GPU, Framework Interop & v1.0

**Goal:** GPU acceleration, framework integration, benchmark suite,
comprehensive documentation, and a stable v1.0 release.

### Deliverables

| # | Deliverable | Module |
| --- | --- | --- |
| 3.1 | PyTorch backend for GPU-accelerated spatial transforms | `backends/torch/` |
| 3.2 | torchvision / MONAI / TorchIO compatibility wrappers | `interop/` |
| 3.3 | Vendor DICOM parsers — Hologic, GE, Siemens | `io/dicom_vendor/` |
| 3.4 | Remaining Phase 2 transforms (motion, scatter, CLAHE, …) | `transforms/modality/` |
| 3.5 | Benchmark suite (speed, memory, augmentation diversity) | `benchmarks/` |
| 3.6 | Sphinx documentation site + API reference | `docs/sphinx/` |
| 3.7 | Jupyter tutorials (MRI, CT, DBT) | `notebooks/` |
| 3.8 | v1.0 release + GitHub Actions CD pipeline | `.github/` |

### Acceptance criteria (Phase 3 / v1.0)

- All Phase 1 and 2 deliverables shipped.
- GPU speedup ≥ 5× vs CPU baseline for spatial transforms.
- All transforms complete in < 500 ms on CPU for a 512×512×80 DBT volume.
- CI/CD publishes wheels to PyPI on tag.

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
