# Project review — 25 October 2026

MedAugmentX has a useful foundation, but it is not ready for a stable medical
imaging release. Its largest weakness is the gap between array-level correctness
and patient-space correctness. Expanding the transform catalog or adding a GPU
backend should follow resolution of the geometry and integration contracts.

This review covers the core, all transform implementations, presets, validation,
serialization, I/O, adapters, tests, examples, notebooks, documentation,
benchmarks, packaging, and CI. Findings below distinguish reproduced defects,
source-level risks, and proposed improvements. No patient datasets, real scanner
exports, or clinical/model-performance experiments were used. Real PyTorch,
MONAI, and TorchIO packages were not installed in the review environment.

The starting checkout was clean at `9bf04b1`, version `0.10.0`. The original
suite passed **619 tests with 92% line coverage**. That did not detect the
CT unit mismatch, DBT axis errors, or several target/metadata problems.

## Improvements made in this review

These changes include focused regression tests. New defect tests were run
against the old implementation and failed before the corresponding fixes.

| Area | Evidence before the fix | Implemented improvement |
| --- | --- | --- |
| CT preset | With only its window and noise steps forced on, seed 0 produced a range of approximately `[-21.91, 20.76]` and standard deviation `6.11` after a `[0, 1]` window | Windowing now clips without normalizing, keeping noise sigma in the native HU scale; examples document normalization as a separate step |
| Affine axis names | A rotation labelled `z` preserved the last array coordinate, while `x` preserved the first | Matrices now respect `(z, y, x)`; DBT uses rotation about `z` for in-plane motion |
| DBT elastic defaults | Preset `alpha=(80, 80, 6)` put a large component on depth | Defaults and teaching examples now put the small component first, e.g. `(6, 80, 80)` |
| Elastic smoothing | Each component used an isotropic scalar sigma, rather than the spatial sigma tuple | Every component is smoothed with the same per-spatial-axis tuple |
| Elastic landmarks | For pull field `d_y(q)=0.5*q_y`, input landmark `y=12` mapped to `6`; the correct answer is `8` | Bounded inverse iteration checks the pull-equation residual and raises on nonconvergence |
| Volume invariants | Empty axes, nonpositive/nonfinite spacing, and nonfinite targets were accepted | Construction rejects these inputs with clear errors |
| Gamma | Scalar `0` collapsed a ramp; scalar `-1` emitted infinity; range validation differed from scalar validation | Both forms require finite, positive, ordered values |
| Low resolution | Small zoom factors on short axes raised `ZeroDivisionError` | Temporary dimensions retain at least one voxel per axis |
| Inspection | A `Guard` hid its wrapped pipeline in a truncated parameter value | Inspection traverses the guarded subtree |
| Documentation | The catalog marked `AnisotropicElastic` as leaving targets unchanged; custom-transform templates omitted seeds | Catalog recognizes the wrapper; templates retain constructor seeds; misleading scientific and reproducibility claims were reduced |

The 100-seed alignment test now compares image and mask exactly under identical
nearest-neighbour interpolation. Previously it checked only whether the mask
centroid sat in bright image foreground. The slice-dropout mask test no longer
uses the tautology `diffs >= 0`.

**Compatibility:** corrected rotations, smoothing, DBT defaults, and CT units
change outputs for existing seeds. Saved JSON describes configuration, not
version-independent behavior. Reproducing old studies requires the old library
and environment. Elastic target inversion can now raise for difficult fields;
it does not guarantee global invertibility or clinical plausibility.

## Remaining release blockers

### 1. Patient-space geometry is incomplete — P1, reproduced

Relevant code: `core/volume.py`, `transforms/spatial/{crop,resize}.py`,
`io/nifti.py`, `io/dicom.py`.

`MedVolume` carries spacing but has no authoritative origin/direction/affine
contract. NIfTI stores its source affine in an otherwise unconstrained metadata
dictionary. Crop, pad, and resize update arrays/targets without updating it;
`save_nifti` reuses the stale matrix. This was already acknowledged in
`COMMERCIAL_ADOPTION.md`; it is an active correctness defect, not just missing
documentation.

Reproduction: a NIfTI of native shape `(10, 12, 14)` and affine diagonal
`(1, 2, 3, 1)`, loaded and center-cropped to `(4, 4, 4)`, retains world origin
`(0, 0, 0)`. The correct origin is `(3, 8, 15)` mm. NIfTI spatial units are
also ignored: a header marked `meter` retains raw zoom values even though
`MedVolume.spacing` promises millimetres.

**Action:** establish one explicit voxel-to-world convention, including array
axis ordering, RAS/LPS, spatial units, affine shear, and unknown geometry.
Compose crop/pad/resize grid mappings into that affine. Specify separately
whether a transform changes the sampling grid or moves anatomy within a fixed
grid. Preserve or deliberately set NIfTI qform/sform and units. Test oblique,
translated, reflected, and sheared inputs using world-coordinate landmarks.
NiBabel's [NIfTI geometry documentation](https://nipy.org/nibabel/nifti_images.html)
describes the distinct affine representations that exports need to handle.

### 2. Resize uses incompatible coordinate and spacing conventions — P1, reproduced

`Resize` uses SciPy's default center-aligned sampling and maps targets with
`(new_size-1)/(old_size-1)`, but computes spacing with `old_size/new_size`.
Resizing a 10-voxel axis at 1 mm to 20 voxels moves endpoint landmark `9` to
index `19` with reported spacing `0.5`: its inferred coordinate changes from
9 mm to 9.5 mm.

**Action:** choose a single center-extent or voxel-extent convention and apply
it to interpolation, landmarks, spacing, origin, and affine. Define singleton
axes explicitly. Add endpoint and interior world-coordinate tests. This is a
behavioral/API decision; changing just the spacing formula would leave other
geometry gaps. See SciPy's [`zoom` grid-mode definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html).

### 3. Real framework integration is not established — P1, source-level finding

`interop/adapters.py` strips a channel dimension but does not explicitly convert
TorchIO spatial layout to MedAugmentX's reversed array convention. It reads
affine/spacing but writes back only tensor data, so resize/crop can leave the
returned framework image geometry stale. MONAI `MetaTensor` affine/history is
not explicitly propagated either. Framework integration tests currently use
NumPy stand-ins, and the default CI matrix installs no framework extras.

**Action:** add optional CI jobs using actual framework objects, anisotropic
non-cubic volumes, and translated/oblique affines. Check physical axis selection,
image/label registration, shape/spacing/affine after grid changes, tensor dtype
and device, source immutability, and missing explicit label keys. Support one
documented convention end to end before claiming patient-space compatibility.
TorchIO's [data model](https://torchio.readthedocs.io/_/downloads/en/latest/pdf/)
includes both a spatial tensor layout and a voxel-to-world affine.

### 4. DICOM loading silently discards important geometry — P1, reproduced/source review

The loader checks projected slice positions, but does not preserve patient
origin or orientation in the returned volume. A reproduced stack with positions
`(0.5*i, 0, i)` is accepted with spacing `(1, 0.5, 0.5)` and no affine, losing
the in-plane displacement between slices. One valid projected spacing is not
enough to describe a tilted/sheared stack.

Single-file multiframe data can also enter a generic branch that assumes one
spacing and rescale without parsing enhanced per-frame functional groups.
Unreadable files are skipped without a report; partial geometry can fall back
to ordinal sorting without telling the caller. Color rejection and irregular
spacing checks are useful but do not establish broad DICOM support.

**Action:** implement an explicit supported-IOD policy; reject unsupported
enhanced/multiframe objects until parsed correctly. Validate row and column
unit vectors and orthogonality, frame of reference, consistent grid vectors,
and per-frame scaling. Preserve the patient-space affine or reject geometry
that cannot be represented. Provide strict loading with skipped-file diagnostics.
Use de-identified fixtures for oblique CT, missing slices, localizers, duplicate
positions, MONOCHROME1, and enhanced DBT. The [DICOM Image Plane Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html)
defines position, orientation, and spacing jointly.

## Other high-value work

### 5. Physical units and task assumptions need explicit policies — P1/P2

`RandomAffine`, elastic deformations, and most blur/artifact strengths operate
in voxels and do not consult spacing. Correct axis labels do not make a
rotation physically rigid on unequal voxel spacing. A column flip also does
not necessarily mean anatomical left/right on an oblique or unreoriented image.
Presets enable flips without remapping left/right class identities or landmark
names. Zero-fill defaults insert water-like values into HU-scale CT backgrounds.

**Action:** offer explicit voxel/mm parameter units, spacing-aware rotation
(`S^-1 R S` in voxel coordinates), dataset-approved flip policies and target-label
remapping, and modality-appropriate fill values. Require callers to choose
normalization and orientation assumptions. Review depth scaling in the DBT
affine preset; in-plane rotation currently still permits per-axis scale changes.

### 6. Nonlinear boxes and deformation topology remain approximate — P1/P2

The inverse landmark fix verifies a local residual. It does not prove a unique
global inverse, exclude folds, or prevent repeated anatomy from reflected
boundaries. Bounding boxes are still mapped through corners only; an interior
point can bulge beyond the returned box. `Guard` cannot detect those failures.

**Action:** add displacement/Jacobian limits and optional topology checks;
document nonlinear box accuracy as a separate capability. Offer dense boundary
sampling or mask-derived bounds for detection tasks. Test interior extrema,
nonconvergence, folds, and large displacements. Avoid universal “mask-safe” or
“targets faithfully mapped” claims without these limits.

### 7. Scientific claims need model-specific evidence — P2

Most artifact transforms are useful appearance approximations. `MetalStreak`
adds a starburst; `BeamHardening` subtracts an image-centered bowl;
`MRIMotion` averages moved images; `LimitedAngleBlur` applies an inverse-angle
Gaussian; `CompressionVariation` rescales one in-plane axis. These are not
validated acquisition/reconstruction simulators. Windowing varies a display
mapping, not the scanner acquisition protocol.

Rician noise has a specific complex-Gaussian magnitude model. Actual MRI noise
depends on coil combination and reconstruction; adding noise to an already
noisy magnitude image is another approximation. See the original
[Rician model](https://pmc.ncbi.nlm.nih.gov/articles/PMC2254141/) and the
[experimental reconstruction/noise study](https://pubmed.ncbi.nlm.nih.gov/18440746/).

**Action:** give every modality transform a model card: equation, input domain,
units, reconstruction assumptions, approximations, citations, intended tasks,
and measured validation. Test Rician background moments against the Rayleigh
limit, noise variance against requested sigma, blur response, and artifact
spatial behavior. Evaluate held-out scanners/sites and task performance before
describing any preset as improving generalization.

### 8. Validation is narrower than its naming suggests — P2

`VolumeValidator` checks image statistics and mask bookkeeping. It does not
validate finite/integer mask values, mutated target coordinates, affine/spacing
coherence, anatomy, or image-mask registration. Foreground loss compares voxel
counts, so a legitimate resolution change can look like tissue loss. Aggregate
counts can also miss substantial loss of one class. `Guard` does not recover
from transform exceptions; its retry policy applies only to returned candidates.

**Action:** separate structural, geometric, intensity, and task-specific rules;
make foreground checks per class and account for voxel volume where appropriate.
Expose attempts, rejection reasons, revert rate, and warnings to experiment
logs. Specify which exceptions are intentionally retriable. Frequent fallback
changes the effective augmentation distribution and should be measurable.

### 9. Parameter validation and numerical behavior are inconsistent — P2

The gamma and volume fixes address concrete cases, but many range helpers
still accept nonfinite values, invalid scales, or silently truncate float sizes
to integers. `AnatomicCrop` does not reject nonpositive sizes. High-order
interpolation can overshoot intensities; downsampling has no explicit antialias
policy. `BrightnessContrast`'s scalar semantics were documented incorrectly.

**Action:** share finite scalar/range, positive size, axis, interpolation, and
clip-bound validators across transforms. Add property tests for accepted input
domains, singleton axes, unsigned/signed integer labels, constant arrays, and
extreme values. Define antialias and overshoot behavior instead of relying on
implicit SciPy defaults. Use `hypot` for Rician magnitude to reduce avoidable
overflow from explicitly squaring large values.

### 10. Reproducibility needs schema and execution provenance — P2

The seed documentation is mostly sound. What is missing is a versioned policy
schema and an execution record of sampled transforms/parameters. Bare class
names and constructor values cannot explain changed behavior across releases.
Copying stateful pipelines to workers can duplicate streams, and shared transform
instances reused in containers can be reseeded by another owner.

**Action:** add schema/package versions, migration rules, policy hashes, and
optional sampled-parameter replay. Offer stable per-sample seeding based on
experiment, epoch, sample identity, and augmentation index; document distributed
rank/worker behavior. Detect reused transform instances or define ownership.
Apply the same finite-value validation to JSON, YAML, and dictionary loading;
JSON numeric overflow such as `1e309` is distinct from the `NaN`/`Infinity`
tokens already rejected by the parser.

### 11. CI measures coverage more than scientific invariants — P2

The existing multi-Python and installed-wheel jobs are strengths. Important
gaps are real framework objects, dependency lower bounds, branch coverage,
OS differences, world-coordinate invariants, and statistical model tests.
Hypothesis is installed but unused. `CONTRIBUTING.md` says formatting is enforced;
CI runs `ruff check`, without `ruff format --check`.
Strict Sphinx builds also depend on three external reference inventories;
unreachable inventories emit warnings and therefore fail under `-W`.

**Action:** prioritize independent oracles: known landmarks under known maps,
round-trip patient coordinates, statistical moments, field gradients,
class-preservation policies, and intentional failure cases. Add a small
minimum-dependency job and optional integration environments. Run lint/type
checks once on a representative Python rather than installing those tools in
every test-matrix job. Introduce a deliberate formatting baseline before
enforcing it. Coverage percentage is a backstop, not a release-quality metric.

### 12. Benchmark and release criteria need measurable definitions — P2/P3

The benchmark reports best-of-N without a machine/environment record, memory,
median/tail latency, or a saved comparison baseline. The advertised CPU target
is acknowledged as unmet, and a future GPU backend does not satisfy a CPU
latency target. Broad performance claims should follow measurements.

**Action:** export benchmark JSON with versions, hardware, threads, shape,
spacing, dtype, mask/target presence, constructor settings, median/p95 latency,
and peak memory. Benchmark representative patches and full volumes separately;
measure adapters and loader-worker throughput as well as leaf transforms.
Prefer explicit supported configurations over one universal 500 ms threshold.

Keep the small core, lazy optional imports, wheel tests, synthetic examples,
and clear license. Add release gates for geometry and real integrations to
`MILESTONES.md`, a single source of version metadata, and trusted publishing
when configured by the maintainer. Keep patient identifiers and full headers
out of public reports; DICOM metadata currently retains patient ID by default.

## Recommended sequence and completion criteria

1. **Geometry correctness:** resolve findings 1–4. Completion requires affine,
   spacing, units, and world landmarks to agree through load → crop/pad/resize
   → export, including oblique and anisotropic inputs. Unsupported DICOM data
   must fail clearly rather than be assigned guessed geometry.
2. **Task/preset contracts:** resolve findings 5–9. Every preset states input
   intensity domain, orientation, units, fill policy, target semantics, and
   limitations. Quantitative phantom/statistical tests support each model.
3. **Research reliability:** resolve findings 10–11. Policies identify their
   schema/environment; actual framework integration and per-sample/worker
   behavior are exercised in CI. A guard's effective rejection rate is logged.
4. **Performance and release:** address finding 12, then invest in optional GPU
   execution and a stable release. Compare against established libraries on
   matched workloads and held-out data before making superiority claims.

## Review validation

Local environment: macOS ARM64, Python 3.11.14, NumPy 2.4.4, SciPy 1.17.1,
NiBabel 5.4.2, pydicom 3.0.2, pytest 9.0.3, Sphinx 8.2.3.

The updated suite passes **652 tests with 92% line coverage**. Ruff and mypy
pass. Wheel and sdist builds and `twine check` pass. The copied suite outside
the checkout passes **547 tests**, with two expected source-only checks skipped.
All three notebooks execute successfully. Eight synthetic example scripts and
2D/3D small-shape benchmark smoke checks pass. The strict Sphinx build succeeds
with network access for its external inventories.
This local run does not establish compatibility across the entire advertised
Python/dependency matrix or clinical validity.
