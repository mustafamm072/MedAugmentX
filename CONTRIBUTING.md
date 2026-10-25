# Contributing to MedAugmentX

Thanks for your interest in MedAugmentX. This guide covers the development
setup, the project conventions, and what we look for in pull requests.

## Development setup

```bash
git clone https://github.com/mustafamm072/MedAugmentX.git
cd MedAugmentX
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
  interop/                  # Framework adapters for dataset samples
  presets.py                # mri_pipeline, ct_pipeline, dxr_pipeline, dbt_pipeline
  serialization.py          # to_json / from_json / to_yaml / from_yaml / REGISTRY
  phantoms.py               # Synthetic MRI/CT/X-ray/DBT volumes for docs and tests
tests/                      # Mirrors the source layout
docs/                       # Architecture, milestones, API reference, API examples
examples/                   # Runnable scripts
notebooks/                  # Modality tutorials (committed without outputs)
```

## Coding conventions

- **Type hints everywhere.** Public functions and classes are fully typed.
- **Numpy first.** Core ops use `numpy` and `scipy.ndimage`. Heavy ML
  frameworks stay optional and must not be imported by core modules.
- **Mask consistency is non-negotiable.** Spatial transforms must use
  nearest-neighbour interpolation for masks and the *same* random sample as
  the image. Add a regression test for any new spatial transform proving this.
- **Seedable RNG.** Never call `np.random` directly inside a transform — use
  `self.rng` from the base class. This is what makes `Compose(seed=...)`
  deterministic.
- **No prints, no globals.** Use `logging` if you really need diagnostics.
- **Style.** `ruff check` and `ruff format` should pass. CI enforces this.
- **Clinical boundaries.** Do not imply diagnostic, treatment, regulatory, or
  clinical decision-support claims in docs, examples, or code comments.
- **Privacy by default.** Examples must use synthetic data or clearly
  de-identified local paths. Do not commit patient identifiers, full DICOM
  headers, access tokens, or private dataset details.

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
            "params": {"strength": self.strength, "p": self.p, "seed": self._seed},
        }
```

`to_dict()` is required for JSON/YAML serialisation. The `"params"` values
must be valid `__init__` keyword arguments so `from_dict()` can reconstruct
the transform. Register the class with `@register_transform` if users will
serialise pipelines that include it:

```python
from medaugmentx.serialization import register_transform

@register_transform
class MyTransform(Transform):
    ...
```

The decorator validates the class and refuses to overwrite an existing name
(pass `override=True` to replace one deliberately). Direct assignment
(`REGISTRY["MyTransform"] = MyTransform`) still works.

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

## Notebooks

Tutorials live in `notebooks/` and are committed **without outputs** — clear
them before committing (*Kernel → Restart Kernel and Clear All Outputs*).
`tests/test_phantoms.py` fails if outputs or execution counts sneak in. CI
executes every notebook on Python 3.12; run them yourself with:

```bash
pip install -e ".[notebooks]"
python notebooks/run_notebooks.py
```

Notebooks must build their data from `medaugmentx.phantoms` — never a download
or a local dataset path, so they run anywhere and never touch patient data.

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

## Reporting security issues

Please do not open public issues for suspected vulnerabilities or accidental
exposure of private medical data. Follow [SECURITY.md](SECURITY.md) and share
only the minimum reproduction needed for maintainers to investigate safely.

## Code of conduct

Be kind, be specific, assume good faith. We follow the
[Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

## Documentation and package checks

```bash
pip install ".[docs]" build twine
python -m sphinx -W --keep-going -b html -c docs/sphinx . /tmp/medaugmentx-docs
python -m build
python -m twine check dist/*
```

CI also installs the wheel in a separate environment and runs the copied tests
outside the source checkout, so editable imports cannot hide packaging errors.

## Releasing

The version number is recorded in five places and they must agree:

| File | Field |
| --- | --- |
| `pyproject.toml` | `project.version` |
| `medaugmentx/__init__.py` | `__version__` |
| `CITATION.cff` | `version`, `date-released` |
| `README.md` | the `print(medaugmentx.__version__)` comment |
| `CHANGELOG.md` | the new `## [x.y.z] — YYYY-MM-DD` heading |

The documentation site reads its version from the installed package, and its
API pages are generated from the source, so nothing under `docs/` needs a
version bump by hand.

Move the `## [Unreleased]` entries under the new heading and leave
`[Unreleased]` empty at the top. Also update the README's Phase-3 status row
and any `docs/MILESTONES.md` deliverable the release completes.

Then tag an annotated tag and publish a GitHub release:

```bash
git tag -a vX.Y.Z -m "MedAugmentX X.Y.Z — <short summary>"
git push origin main && git push origin vX.Y.Z
gh release create vX.Y.Z --title "vX.Y.Z — <Summary>" --notes-file <notes>
```

Release notes follow the shape of the previous entries: a `## MedAugmentX
X.Y.Z — <Title>` heading, a short intro, `### Highlights` with bold-lead
bullets, per-area sections, compatibility notes, then install and
full-changelog compare links.

Note that CI's oldest matrix entry (Python 3.9) resolves `pydicom` 2.x, the
last series that supports it. Test fixtures and I/O code must work on both
`pydicom` 2.x and 3.x.
