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
| [`dbt_pipeline.py`](dbt_pipeline.py) | The tomosynthesis pipeline on a synthetic DBT slab with anisotropic spacing. |
| [`framework_interop.py`](framework_interop.py) | Use `TorchTransform` and `MonaiMapTransform` with dict samples. |
| [`custom_transform.py`](custom_transform.py) | How to author your own seedable transform and drop it into `Compose`. |
| [`load_and_augment.py`](load_and_augment.py) | Load a real NIfTI / DICOM volume from disk, augment, and write back. Requires the `io` extra. |

If the optional I/O backends are not installed, the loader scripts fail
fast with a clear `ImportError` telling you which extra to install.
