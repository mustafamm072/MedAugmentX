# Migrating to 1.0.0

Version 1 establishes the public NumPy/SciPy API and changes numerical behavior.
Keep the original environment when reproducing pre-1.0 studies. Saved policies
are configurations; they do not make old numerical behavior portable.

## Patient coordinates and resize

`MedVolume.affine` is a finite nonsingular 4×4 matrix mapping **array indices**
`(z, y, x)` or `(y, x)` to RAS **millimetres**. Its first `ndim` columns define
the grid basis; its last column is the origin. `spacing` is the length of those
basis vectors. With no affine, patient geometry is unknown, even if spacing is
provided. Array x is a column direction, not automatically anatomical left/right.
Legacy `metadata['affine']` is interpreted as native XYZ-to-RAS and migrated;
new code should use the field. Do not mutate spacing independently of affine.

Crop and pad translate the output grid. Resize preserves the full voxel extent:
`q = (p + 0.5) * new_size / old_size - 0.5` for targets and
`p = (q + 0.5) * old_size / new_size - 0.5` for image sampling. This convention
also handles singleton axes and matches SciPy `grid_mode=True`. Downsampling
anti-aliasing is enabled for images; masks always use nearest-neighbour sampling.
Active affine/elastic/flip augmentation moves anatomy on the existing grid.

NIfTI loads reverse native spatial axes and convert metre/micron headers to mm.
Loaded arrays own their data and release source-file memory mappings, including
on Windows. Loading materializes the image and optional mask in memory.
Image and `mask_path` must share a patient grid. Export updates sform and qform;
shear is retained in sform with qform disabled. Unknown-grid export warns;
set `allow_assumed_geometry=False` to require known patient geometry.

## Geometry and target behavior

`RandomAffine(space='physical')` now preserves physical distances for rotations
without scaling, including anisotropic/sheared grids. Its angles are in an
orthonormal image frame, not RAS anatomical axes. `space='voxel'` selects the
legacy index metric. Translation units can be fraction, voxels, or mm along grid
axes. DBT presets scale only in-plane by default.

Elastic fields now limit sampled displacement gradients by default. Optional
`min_jacobian` rejects bad sampled determinants. These numerical checks do not
establish anatomical realism or global continuous invertibility. Targets use
iterative pull-field inversion and raise if it fails. Boxes use conservative
global displacement bounds by default; these can be loose. Choose
`bbox_mode='corners'` for the previous approximation or `'reject'` to disallow
boxes. Conservative boxes are computed directly without corner inversion.
For constant image padding, inverse target mapping extends the displacement
field with its nearest boundary value; image/background fill remains constant. `units='mm'` converts component/smoothing lengths along each grid axis;
separable filtering on a sheared grid is not isotropic in RAS. Affine/elastic
image and label resampling share the chosen boundary-extension mode, with
constant label fill always zero; reflected image context carries reflected labels.

## Presets and input domains

Presets disable flips unless `allow_flip=True`. Decide whether reversal is valid
for your task, then explicitly remap laterality classes if needed:
`RandomFlip(label_map={1: 2, 2: 1})`; landmark/box label maps are also supported.
Mappings must be involutions. Array orientation alone cannot identify laterality.

MRI, DXR and DBT appearance parameters expect nonnegative, approximately `[0,1]`
inputs. Use their `normalize=True` option for per-volume percentile clipping and
normalization. This can discard absolute intensity information; use training-set
preprocessing instead when that information matters. CT presets take HU, retain
HU internally and use air fill `-1000`; `normalize_output=True` explicitly windows
the final result to `[0,1]`. Presets are starting policies, not tuned defaults.
Rician noise is a single-complex-signal magnitude perturbation; modality effects
are documented approximations, not scanner simulators.

`VolumeValidator.max_foreground_loss` measures each nonzero class in physical
area/volume by default. Choose `foreground_measure='voxels'` for count-based
behavior. Guard statistics expose attempts, rejections, reverts and reasons.
`catch_errors=True` treats ValueError/FloatingPointError draws as failures;
programming errors propagate. Monitor fallback rates rather than silently
assuming every training image received augmentation.

## Policies, worker seeding and replay

JSON/YAML writers now emit schema/library versions and a canonical policy hash.
Loaders still accept legacy documents with a warning; review and re-export them.
Nonfinite values, invalid sizes/ranges and cyclic documents fail early.
Histogram references serialize compact quantile summaries instead of image arrays.

A transform instance cannot occupy multiple pipeline positions or parents.
Construct a fresh instance or deepcopy it. Container seeds still override child
streams. `reseed(seed)` restarts a policy. `apply_for_sample` derives a stable
seed from experiment seed, sample ID, epoch and augmentation index, independent
of worker assignment. Explicitly pass the epoch to worker copies, including
persistent workers; updating only the parent dataset does not update them.

`record_augmentation` stores exact RNG checkpoints, branch decisions, policy,
environment and input/output fingerprints. `replay_augmentation` verifies the
same environment and input before replay. Records contain no input image or
metadata, but policies may describe data distributions. Review artifacts before
sharing. Hashes detect changes and provide no authenticity guarantee. Custom
transforms need registered constructors and serializable configuration; hidden
state outside the RNG is not replayed. See `examples/preview_and_replay.py`.

## Integration limits

PyTorch tensor processing is CPU NumPy/SciPy processing and is not differentiable.
Adapters support single-channel images with an optional registered label image.
MONAI/TorchIO use native XYZ spatial order by default; plain mapping adapters
use array order unless configured. Current MetaTensor affine is propagated;
pre-existing MONAI inversion history is cleared because these operations do not
provide MONAI inverse transforms. Missing configured labels, additional TorchIO
images, and mismatched grids raise rather than silently desynchronize samples.

DICOM strict loading supports classic monochrome single-frame CT/MR/CR/DX/MG
objects with coherent geometry. It preserves regular shear/gantry tilt. Enhanced,
multiframe, irregular stacks, functional groups, real-world/LUT mappings and
unsupported SOP classes raise. Decode codecs must be installed for compressed
pixel data. `strict=False` can skip unreadable/non-image candidates, with warnings
and `skipped_files` diagnostics; it does not override incompatible known geometry.
PatientID is omitted unless `include_patient_id=True`. Selected source paths and
UIDs remain metadata; this loader does not de-identify a dataset.
