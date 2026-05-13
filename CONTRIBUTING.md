# Contributing to MedAugmentX

Thanks for your interest in MedAugmentX. This guide covers the development
setup, the project conventions, and what we look for in pull requests.

## Development setup

```bash
git clone https://github.com/medaugmentx/medaugmentx.git
cd medaugmentx
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The `dev` extra pulls in `pytest`, `hypothesis`, `pydicom`, `nibabel`, `ruff`,
and `mypy`.

## Repository layout

```
medaugmentx/                 # Library source
  core/                     # MedVolume, Transform base, Compose/OneOf/SomeOf
  transforms/               # Spatial, intensity, modality-specific transforms
    spatial/                # RandomAffine, RandomFlip, ElasticDeform, AnatomicCrop
    intensity/              # GaussianNoise, BiasField, WindowLevel, …
    modality/mri/           # GhostingArtifact, KSpaceDropout
    modality/ct/            # BeamHardening
    modality/tomosynthesis/ # SlabShift, LimitedAngleBlur, SliceDropout, …
  io/                       # DICOM, NIfTI loaders (optional dependency)
  presets.py                # mri_pipeline, ct_pipeline, dxr_pipeline, dbt_pipeline
  serialization.py          # to_json / from_json / to_yaml / from_yaml / REGISTRY
tests/                      # Mirrors the source layout
docs/                       # Architecture, milestones, API examples
examples/                   # Runnable scripts
```

## Coding conventions

- **Type hints everywhere.** Public functions and classes are fully typed.
- **Numpy first.** Core ops use `numpy` and `scipy.ndimage`. Heavy ML
  frameworks are reserved for Phase 3 backends and stay optional.
- **Mask consistency is non-negotiable.** Spatial transforms must use
  nearest-neighbour interpolation for masks and the *same* random sample as
  the image. Add a regression test for any new spatial transform proving this.
- **Seedable RNG.** Never call `np.random` directly inside a transform — use
  `self.rng` from the base class. This is what makes `Compose(seed=...)`
  deterministic.
- **No prints, no globals.** Use `logging` if you really need diagnostics.
- **Style.** `ruff check` and `ruff format` should pass. CI enforces this.

## Adding a new transform

Use this template:

```python
from __future__ import annotations
from typing import Any
from medaugmentx.core import Transform, MedVolume

class MyTransform(Transform):
    """One-line summary.

    Longer description: what physical / clinical effect this simulates,
    and any modality assumptions.
    """

    def __init__(self, strength: float = 0.1, p: float = 1.0, seed=None):
        super().__init__(p=p, seed=seed)
        self.strength = float(strength)

    def apply(self, volume: MedVolume) -> MedVolume:
        # Sample parameters from self.rng, never np.random
        delta = self.rng.uniform(-self.strength, self.strength)
        new_image = volume.image + delta
        # If the transform is spatial, also transform volume.mask with
        # nearest-neighbour interpolation using the *same* sampled params.
        return volume.replace(image=new_image)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "params": {"strength": self.strength, "p": self.p},
        }
```

`to_dict()` is required for JSON/YAML serialisation. The `"params"` values
must be valid `__init__` keyword arguments so `from_dict()` can reconstruct
the transform. Register the class in `REGISTRY` if users will serialise
pipelines that include it:

```python
from medaugmentx.serialization import REGISTRY
REGISTRY["MyTransform"] = MyTransform
```

Built-in transforms are registered automatically in `serialization.py`.

Then:

1. Add the class to the appropriate `medaugmentx/transforms/<group>/` module.
2. Re-export it from `medaugmentx/transforms/__init__.py`.
3. Register it in `REGISTRY` inside `medaugmentx/serialization.py` (`_register_builtins`).
4. Add tests in `tests/transforms/<group>/test_<name>.py` covering:
   - output shape and dtype,
   - mask/image consistency (if spatial),
   - reproducibility under a fixed seed,
   - the `p=0` and `p=1` boundary behaviour,
   - JSON round-trip via `to_dict()` / `from_dict()`.

## Tests

```bash
pytest                                   # full suite
pytest tests/core                        # one module
pytest -m "not slow"                     # skip slow tests
pytest --cov=medaugmentx --cov-report=term-missing
```

We aim for ≥ 90% coverage on `medaugmentx/core` and `medaugmentx/transforms`.
I/O tests are marked `@pytest.mark.io` and skipped automatically when the
optional dependency (`pydicom`, `nibabel`) is not installed.

## Commit messages

Short imperative subject (≤ 72 chars). Body explains *why*, not what.

```
Add SlabShift transform for DBT pipelines

Simulates Z-axis reconstruction-centre variation between scans on the
same patient — important for cross-study robustness.
```

## Pull requests

- Open against `main`.
- Include a short rationale and link to the milestone deliverable.
- New transforms must include tests and a docstring.
- CI must be green (pytest + ruff + mypy).

## Reporting bugs

Please include:

- MedAugmentX version (`pip show medaugmentx`),
- a minimal `MedVolume` (shape, dtype, spacing) that reproduces the issue,
- the full traceback,
- modality and, if known, the originating scanner vendor.

## Code of conduct

Be kind, be specific, assume good faith. We follow the
[Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
