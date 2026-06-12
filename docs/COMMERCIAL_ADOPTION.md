# Commercial Adoption

MedAugmentX is built for teams that need medical-image augmentation to be
simple, inspectable, and easy to validate. The core package stays lightweight,
while integrations for common training stacks remain optional.

This guide is a practical checklist for adopting MedAugmentX in research,
commercial model training, and internal ML platforms.

---

## Intended use

MedAugmentX provides data augmentation utilities for medical imaging AI
training and evaluation workflows. It is not a diagnostic device, clinical
decision-support system, image acquisition system, PACS, or regulated medical
device on its own.

Before using augmented data in a regulated or customer-facing product, teams
should document:

- the model task and modality,
- the source data and inclusion/exclusion criteria,
- the augmentation policy and rationale,
- the validation protocol for original and augmented data,
- known limitations and failure modes.

---

## Adoption checklist

| Area | What to verify |
| --- | --- |
| Installation | Core install pulls only `numpy` and `scipy`; optional extras are installed only when needed. |
| Reproducibility | Pipelines use explicit seeds for experiments and are saved with JSON or YAML serialisation. |
| Clinical plausibility | Transform ranges are reviewed by imaging, clinical, or modality experts for the target use case. |
| Labels and masks | Spatial transforms preserve mask alignment and use nearest-neighbour mask interpolation. |
| Data privacy | Patient identifiers are removed or controlled before data enters training or examples. |
| Auditability | Pipeline configs, package version, dataset version, and training code commit are recorded together. |
| Framework fit | Use `TorchTransform`, `MonaiMapTransform`, or `TorchIOTransform` only when those frameworks are already part of the stack. |
| Release control | Pin `medaugmentx` and optional dependency versions for production training runs. |

---

## Dependency policy

The default package is intentionally small:

```bash
pip install medaugmentx
```

Required dependencies:

- `numpy`
- `scipy`

Optional extras:

| Extra | Use |
| --- | --- |
| `io` | DICOM and NIfTI loading/writing through `pydicom` and `nibabel` |
| `yaml` | YAML pipeline serialisation |
| `torch` | PyTorch tensor samples |
| `monai` | MONAI-style dictionary samples |
| `torchio` | TorchIO `Subject` samples |
| `frameworks` | PyTorch, MONAI, and TorchIO integrations together |

Interop modules are import-lazy. Importing `medaugmentx.interop` does not
import PyTorch, MONAI, or TorchIO until framework objects are passed at
runtime.

---

## Reproducibility and audit trail

For production-grade experiments, save the augmentation pipeline next to model
and dataset metadata:

```python
from medaugmentx.presets import mri_pipeline
from medaugmentx.serialization import to_json

pipeline = mri_pipeline(seed=42)
pipeline_json = to_json(pipeline)
```

Record at least:

- `medaugmentx.__version__`,
- pipeline JSON or YAML,
- random seed policy,
- dataset version or manifest,
- model training code commit,
- optional dependency versions for framework integrations.

For per-epoch reproducibility, create a fresh seeded pipeline per epoch rather
than reusing one pipeline instance forever, because RNG state advances with
each sample.

---

## Clinical validation guidance

MedAugmentX provides clinically-aware primitives, but safe transform ranges are
task-specific. A CT lung nodule classifier, MRI segmentation model, and DBT
screening model may require very different augmentation policies.

Recommended validation steps:

1. Review transform ranges with modality experts.
2. Visualize augmented samples across edge cases and scanner/site subgroups.
3. Confirm segmentation masks remain aligned after spatial transforms.
4. Compare model performance on unaugmented holdout data and clinically
   relevant subgroups.
5. Track whether augmentation changes calibration, sensitivity, specificity,
   or clinically meaningful operating points.

Avoid using augmentation to simulate pathology, device behavior, scanner
vendors, or reconstruction algorithms unless the simulation has been validated
for that use case.

---

## Privacy and security

MedAugmentX does not require network access at runtime and does not send image
data anywhere. The library operates on arrays and metadata supplied by the
caller.

Teams remain responsible for:

- de-identifying protected health information before sharing data or examples,
- controlling patient identifiers in `MedVolume.metadata`,
- securing local training environments and artifact storage,
- reviewing optional dependency licenses and security posture.

See [SECURITY.md](../SECURITY.md) for vulnerability reporting and supported
security expectations.

---

## Recommended production pattern

```python
from medaugmentx.io import load_nifti
from medaugmentx.presets import mri_pipeline
from medaugmentx.serialization import to_json

pipeline = mri_pipeline(seed=42)
pipeline_config = to_json(pipeline)

volume = load_nifti("case_001.nii.gz")
augmented = pipeline(volume)
```

Keep `pipeline_config` with training metadata so any model run can be
reconstructed later.
