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

## Reproducible experiment record

Run `python examples/reproducible_experiment.py` to replay a saved MRI policy
on synthetic data and print an experiment record with configuration and output
fingerprints. See [Research guide](../docs/RESEARCH_GUIDE.md).


## README transformation gallery

Generate all four synthetic modality inputs and the eight transformed outputs:

```bash
pip install -e . matplotlib
python examples/generate_readme_gallery.py
```

The script writes `docs/assets/modality-gallery.png` and a JSON manifest with
parameters, display limits, and dependency versions. Use `--output-dir PATH`
for a separate preview. Matplotlib is only needed to regenerate the gallery;
it is not a core library dependency. The inputs are mathematical phantoms,
not patient data or validated anatomical simulations. Effects are intentionally
visible and are not a suggested augmentation policy.

MRI, CT, and X-ray use 2D inputs. Tomosynthesis uses a 3D phantom with spacing
`(1.0, 0.5, 0.25)` and displays its middle-y `(z, x)` plane in physical aspect
ratio. The display range is fixed within each row, and each transform receives
the original input independently.

## Preview and exact replay

Install `medaugmentx[viz]`, then run `python examples/preview_and_replay.py --output preview-output`.
This writes a synthetic image/mask montage, structural QA, versioned policy and
verified replay record. For real NIfTI use `load_nifti(image_path, mask_path=mask_path)`
and keep output artifacts inside your controlled data store.
