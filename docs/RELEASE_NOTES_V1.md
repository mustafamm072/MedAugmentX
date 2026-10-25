# MedAugmentX 1.0.0

MedAugmentX 1.0.0 establishes the public CPU augmentation API for medical-image
AI datasets. This release focuses on patient-space correctness, registered
annotations, reproducible training pipelines and practical inspection tools.
The core install continues to require only NumPy and SciPy.

## Install

```bash
pip install --upgrade medaugmentx==1.0.0
pip install 'medaugmentx[io,viz]==1.0.0'          # NIfTI/DICOM and visual QA
pip install 'medaugmentx[frameworks]==1.0.0'      # PyTorch, MONAI and TorchIO
```

Python 3.9–3.14 is supported. Frameworks, I/O, plotting and YAML remain optional.

## Geometry and annotation correctness

- An authoritative `MedVolume.affine` maps array indices to RAS millimetres.
  Crop, pad and resize update the grid, including oblique and sheared geometry.
- NIfTI image/label loading checks registration, converts spatial units and
  exports coherent sform/qform information. Loaded arrays own their data,
  avoiding source-file locks on Windows.
- Resize uses one voxel-extent convention for image sampling, spacing and
  geometric targets, including singleton axes. Image downsampling anti-aliasing
  is enabled by default; labels use nearest-neighbour interpolation.
- Affine rotations default to the physical grid metric. Elastic deformations
  support displacement/gradient limits, optional Jacobian rejection and
  conservative bounding boxes. Constant image fill is independent of the field
  extension used to invert boundary targets.
- Classic monochrome single-frame DICOM loading checks SOP classes, rescaling,
  orientation and slice geometry, preserves regular shear and reports skipped
  candidates explicitly. PatientID retention is opt-in.

## Reproducibility and training integration

- JSON/YAML policies include schema/library versions and a canonical policy hash.
- `apply_for_sample` derives stable sample/epoch seeds independent of worker
  assignment; `record_augmentation` and `replay_augmentation` save and verify RNG
  checkpoints, branch decisions and input/output fingerprints.
- Real PyTorch, MONAI MetaTensor and TorchIO Subject integrations preserve
  supported geometry and labels. Integration tests include spawned workers.
- Shared transform instances in multiple pipeline positions are rejected.
  Construct a fresh instance or copy the policy when independent state is needed.

## Inspection and safer policy defaults

- `preview_augmentation` creates local image/mask montages and QA reports.
  `model_card` exposes packaged assumptions, equations, units and limitations for
  all 34 transforms.
- `NormalizeIntensity` provides explicit percentile normalization. MRI, DXR and
  DBT presets offer normalization; CT retains HU internally with optional final
  window normalization and appropriate air fill.
- Presets disable flips unless explicitly enabled. Involutive label maps support
  task-specific laterality remapping; DBT scaling defaults to in-plane axes.
- Validation checks geometry, finite/integer mask values and per-class physical
  foreground loss. Guard telemetry reports rejected draws, retries and reverts.
- Constructor domain checks, stable Rician magnitude computation, bounded bias
  fields and compact histogram summaries address numerical and storage risks.

## Upgrade from 0.x

This release changes numerical output for existing policies and seeds. Review the
[v1 migration guide](https://github.com/mustafamm072/MedAugmentX/blob/v1.0.0/docs/MIGRATION_V1.md)
before upgrading. In particular:

- Resize coordinates preserve voxel extent; affine rotations use the physical
  metric unless `space='voxel'` is selected.
- Flips require preset opt-in, elastic gradients are limited by default, and
  foreground loss is measured per class in physical area/volume by default.
- Legacy unversioned policies load with warnings and should be reviewed and
  re-exported. Exact replay requires the recorded software environment.
- MONAI inversion history is cleared because these transforms do not implement
  MONAI inverse operations. Enhanced/multiframe DICOM and incompatible grids
  fail explicitly.

## Verification and research evidence

The release process requires a successful complete CI run for the exact release
commit before uploading to PyPI. CI covers Python 3.9–3.14, representative Linux,
Windows and macOS jobs, minimum dependencies, real frameworks, branch coverage,
installed-wheel tests, documentation, examples and notebooks.

Measured JSON benchmarks record latency, memory and environment metadata. A
fixed-split synthetic segmentation ablation is included with its negative result:
the tested appearance policy reduced Dice relative to no augmentation. Presets
are starting configurations and require held-out evaluation on the target task.

## Scope and limitations

This is a research/training augmentation library. Its modality effects are
documented approximations, not validated scanner simulations or evidence of
clinical benefit. No patient-cohort or external-site model-benefit validation is
claimed. The numerical geometry checks do not establish anatomical plausibility.
There is no GPU backend in this release; tensor adapters run through CPU
NumPy/SciPy processing and are not differentiable.

See the [model cards](https://github.com/mustafamm072/MedAugmentX/blob/v1.0.0/docs/MODEL_CARDS.md),
[release evidence](https://github.com/mustafamm072/MedAugmentX/blob/v1.0.0/docs/RELEASE_READINESS.md)
and [full changelog](https://github.com/mustafamm072/MedAugmentX/blob/v1.0.0/CHANGELOG.md).
