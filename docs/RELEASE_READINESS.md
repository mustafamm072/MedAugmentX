# Version 1.0.0 release readiness

This is the local release candidate prepared from review-fix commit `002e3be`.
Version 1 freezes the documented CPU augmentation API; it does not certify
clinical realism, model benefit, universal scanner support or GPU performance.
See [migration](MIGRATION_V1.md) before upgrading a pre-1.0 policy.

## Review resolution

| October review area | Implemented change and evidence |
| --- | --- |
| Patient geometry | Authoritative array-index-to-RAS-mm affine; oblique/sheared crop/pad/resize and registered NIfTI label round trips |
| Resize | Voxel-extent sampling and target maps, including singleton property tests and anti-aliasing |
| Real integrations | Actual PyTorch, MONAI MetaTensor and TorchIO Subject tests; spawned zero/two-worker deterministic sample comparison |
| DICOM | Supported classic monochrome SOP allowlist, geometry/rescale checks, regular shear, explicit unsupported-object rejection, skip diagnostics and opt-in PatientID |
| Task/preset contracts | Explicit normalization, HU-safe CT fill/order, physical affine units, in-plane DBT scaling, opt-in flips and laterality remapping |
| Deformation safety | Sampled gradient limit, optional Jacobian threshold/cap, inverse residual checks, conservative nonlinear boxes |
| Model assumptions | Packaged/queryable cards for all 34 transforms; numerical noise-moment, filter-response and bias-bound tests |
| Validation | Geometry/target/mask rules, per-class physical foreground loss, Guard retry/revert/exception counters and reasons |
| Numerical domains | Shared finite/range/size checks, stable magnitude hypot, anti-alias/overshoot options, compact histogram quantiles |
| Provenance | Schema/library/hash policies, sample/epoch seeding, RNG checkpoint replay, input/output hashes, ownership rejection |
| CI | Format/type/lint checks, branch coverage, minimum core/framework dependencies, Python/OS jobs, wheel tests, offline docs |
| Research/release resources | JSON environment/latency/memory benchmarks, fixed-split repeated-seed synthetic segmentation ablation, single version source and explicit release-event workflow |

These rows resolve repository implementation defects or bound the supported API.
They do not imply that every modality approximation has been validated against
real acquisition physics. Cards state each model's evidence level and limitations.

The final review regression is fixed: constant image padding no longer causes
boundary-target inversion oscillation, and conservative boxes bypass corner
inversion. Five focused regression cases failed before the fix and now pass.

## Measured evidence

Local checks passed on macOS arm64:

- **719 tests passed**, one optional-framework module skipped in the core env;
  **87.09% combined line/branch coverage** (85% gate).
- **19 real framework adapter tests passed**, including spawned workers, with
  PyTorch 2.14.1, MONAI 1.6.1 and TorchIO 1.2.1.
- Minimum NumPy 1.22.4 / SciPy 1.9.3 / nibabel 4.0.2 / pydicom 2.3.1 were
  exercised on Python 3.10.20 (**718 passed**, two optional modules/tests skipped).
  The installed-wheel run passed **610 tests**
  outside the checkout; optional plotting/framework and source-only docs tests
  skipped. This is not a local Python 3.9 or minimum-framework-version run.
- Ruff lint/format, mypy, strict offline Sphinx build, three notebooks, eight
  standalone examples, preview/JSON replay and MRI/CT CLI image/mask round trips.
- Wheel/sdist build and Twine metadata validation; all 34 model cards are present
  and readable from the installed wheel.

Machine-readable counts, environments and scope are recorded in
`resources/validation/release-checks.json` in the source distribution.
The configured GitHub OS/Python matrix must run on the pushed commit; local
macOS checks do not establish remote Linux/Windows results.

The [benchmark/evaluation guide](../benchmarks/README.md) describes exact
measurement scope. Raw JSON runs are in `resources/validation/` in the source
distribution. The controlled synthetic comparison showed that the tested
appearance augmentation reduced Dice. Keep this negative result when deciding
what to evaluate on a real dataset; the presets are not proven recommendations.

## Publication steps and remaining external evidence

1. Push this candidate and require the complete GitHub CI matrix to pass.
2. Review the migration guide and final diff; freeze the public API for v1.
3. Configure PyPI trusted publishing for this repository's `release.yml` and a
   protected GitHub `pypi` environment. See the [official PyPI setup](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
4. Create tag `v1.0.0` only on the verified commit and explicitly publish its
   GitHub release. The workflow verifies tag/version, tests/builds and uploads
   distributions using OIDC. A normal push does not publish a package.
5. Confirm the installed PyPI artifact, documentation version and archive DOI
   refer to the release before citing it. The existing DOI is not asserted to
   identify this new release.

No release/tag/PyPI publication is performed by preparing these files. There
are no patient cohorts, external-site results, vendor scanner files or calibrated
acquisition simulations in the local evidence. Those require controlled data
and domain review before stronger task-specific claims. GPU support remains
future work. Unsupported DICOM formats fail explicitly rather than being
advertised as supported.
