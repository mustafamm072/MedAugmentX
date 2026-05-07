# Milestones & Roadmap

MedAugment ships in three phases. Phase 1 (this release) is the MVP that
unlocks early-adopter usage; Phases 2 and 3 build out modality coverage
and production polish.

---

## Phase 1 — Core MVP ✅

**Goal:** A pip-installable library with the most essential transforms and
DICOM/NIfTI I/O. Enough for early adopters and internal testing.

### Deliverables

| # | Deliverable | Module | Status |
| --- | --- | --- | --- |
| 1.1 | `MedVolume` dataclass + `Transform` ABC | `medaugment/core/{volume,base}.py` | ✅ |
| 1.2 | `Compose`, `OneOf`, `SomeOf` | `medaugment/core/compose.py` | ✅ |
| 1.3 | `RandomAffine`, `RandomFlip`, `AnatomicCrop` (3D) | `medaugment/transforms/spatial/` | ✅ |
| 1.4 | `ElasticDeform` with anisotropic sigma | `medaugment/transforms/spatial/elastic.py` | ✅ |
| 1.5 | `RicianNoise`, `GaussianNoise`, `GammaCorrection` | `medaugment/transforms/intensity/` | ✅ |
| 1.6 | DICOM loader with spacing metadata | `medaugment/io/dicom.py` | ✅ |
| 1.7 | NIfTI loader/writer | `medaugment/io/nifti.py` | ✅ |
| 1.8 | `SlabShift` + `LimitedAngleBlur` (DBT) | `medaugment/transforms/modality/tomosynthesis/` | ✅ |
| 1.9 | `SliceDropout`, `AnisotropicElastic` (DBT) | `medaugment/transforms/modality/tomosynthesis/` | ✅ |
| 1.10 | Unit + integration test suite (pytest) | `tests/` | ✅ |
| 1.11 | PyPI packaging + README docs | `pyproject.toml`, `README.md` | ✅ |

### Acceptance criteria (Phase 1)

- All deliverables above merged.
- `pytest` is green on every supported Python version in CI.
- A user can `pip install medaugment` and run the README quick-start without
  installing PyTorch / MONAI / TorchIO.
- Identical seeds produce identical output across runs (regression test in
  `tests/integration/test_full_pipeline.py`).

---

## Phase 2 — Medical-Specific Transforms

**Goal:** Full modality coverage including physics-based artifact
simulation, the complete DBT module, and validated preset pipelines for
all four modalities.

### Deliverables

| # | Deliverable | Module |
| --- | --- | --- |
| 2.1 | `BiasField` (N4-style), `MRIGhosting`, `KSpaceCorruption` | `transforms/modality/mri.py` |
| 2.2 | `MRIMotion` (in-plane rigid motion blur) | `transforms/modality/mri.py` |
| 2.3 | `BeamHardening`, `MetalStreak` | `transforms/modality/ct.py` |
| 2.4 | `HUWindowShift`, `CLAHEContrast`, `HistogramMatch` | `transforms/intensity/` |
| 2.5 | `ScatterSimulation`, `GridArtifact` | `transforms/modality/xray.py` |
| 2.6 | `CompressionVariation`, `ReconStreak` | `transforms/modality/tomosynthesis/` |
| 2.7 | Vendor DICOM parsers — Hologic, GE, Siemens | `io/dicom_vendor/` |
| 2.8 | Preset pipelines: `MRI_STANDARD`, `CT_STANDARD`, `DBT_STANDARD` | `pipelines/presets.py` |
| 2.9 | Anatomical plausibility validator | `validators/plausibility.py` |
| 2.10 | YAML serialisation / deserialisation of pipelines | `pipelines/serial.py` |

### Acceptance criteria (Phase 2)

- ≥ 90% test coverage on `transforms/modality/`.
- DICOM round-trip preserves spacing within 0.001 mm for all four vendors.
- Preset pipelines pass radiologist-defined plausibility checks (manual
  review panel).

---

## Phase 3 — Advanced & Production-Ready

**Goal:** GPU acceleration, framework integration, benchmark suite,
comprehensive documentation, and a stable v1.0 release.

### Deliverables

| # | Deliverable | Module |
| --- | --- | --- |
| 3.1 | PyTorch backend for GPU-accelerated spatial transforms | `backends/torch/` |
| 3.2 | SynMed integration hook | `pipelines/synmed.py` |
| 3.3 | torchvision / MONAI compatibility wrappers | `interop/` |
| 3.4 | Benchmark suite (speed, memory, augmentation diversity) | `benchmarks/` |
| 3.5 | Sphinx documentation site + API reference | `docs/sphinx/` |
| 3.6 | Jupyter tutorials (MRI, CT, DBT) | `notebooks/` |
| 3.7 | v1.0 release + GitHub Actions CD pipeline | `.github/` |

### Acceptance criteria (Phase 3 / v1.0)

- All Phase 1 + 2 deliverables shipped.
- GPU speedup ≥ 5× vs CPU baseline for spatial transforms.
- All transforms execute in < 500 ms on CPU for a 512×512×80 DBT volume.
- CI/CD publishes wheels to PyPI on tag.
- Sustained PyPI download volume and external academic citations as
  community-adoption signals.

---

## How to track progress

Issues and pull requests use the labels below. Phases map 1:1 to GitHub
milestones (`phase-1`, `phase-2`, `phase-3`).

| Label | Use |
| --- | --- |
| `transform` | New augmentation primitive |
| `io` | DICOM / NIfTI / new format |
| `dbt` | Tomosynthesis-specific |
| `bug` | Defect |
| `docs` | Documentation only |
| `phase-2`, `phase-3` | Roadmap milestones |

Contributors: see [CONTRIBUTING.md](../CONTRIBUTING.md).
