# Examples

Self-contained scripts that exercise the public MedAugmentX surface. From a
source checkout, install the package with `pip install -e .` or run with
`PYTHONPATH=.`:

```bash
PYTHONPATH=. python examples/<name>.py
```

| Script | What it shows |
| --- | --- |
| [`quickstart.py`](quickstart.py) | The MedAugmentX "hello world" — build a `MedVolume`, run a mixed pipeline, inspect the result. |
| [`new_transforms.py`](new_transforms.py) | Tour of the 0.6.0 additions: cutout, shape normalisation, CLAHE, histogram matching, and the new MRI/CT/X-ray/DBT artifacts. |
| [`dbt_pipeline.py`](dbt_pipeline.py) | The tomosynthesis pipeline on a synthetic DBT slab with anisotropic spacing. |
| [`framework_interop.py`](framework_interop.py) | Use `TorchTransform`, `MonaiMapTransform`, and `TorchIOTransform` with framework-style samples. |
| [`custom_transform.py`](custom_transform.py) | How to author your own seedable transform and drop it into `Compose`. |
| [`safe_augmentation.py`](safe_augmentation.py) | Validate augmented volumes with `VolumeValidator` and wrap a pipeline in `Guard` (raise / warn / revert / retry). |
| [`keypoints_bboxes.py`](keypoints_bboxes.py) | Track landmark keypoints and bounding boxes through a spatial pipeline, then prune off-frame targets after a crop. |
| [`load_and_augment.py`](load_and_augment.py) | Load a real NIfTI / DICOM volume from disk, augment, and write back. Requires the `io` extra. |

If the optional I/O backends are not installed, the loader scripts fail
fast with a clear `ImportError` telling you which extra to install.
