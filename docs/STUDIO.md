# Policy Studio

The Policy Studio is a local web application for designing, previewing and
stress-testing augmentation policies on your own scans before you train with
them. It ships with the library, adds no dependencies, and exports exactly the
JSON, YAML or Python your training code loads.

```bash
medaugmentx studio --phantom mri                       # synthetic data
medaugmentx studio --image t1.nii.gz --mask seg.nii.gz # your scan and labels
medaugmentx studio --image /data/ct/series01/          # a DICOM series
python -m medaugmentx.studio --policy policy.json      # continue editing a policy
```

The command prints an address such as `http://127.0.0.1:53817/?token=…` and
opens it in your default browser. Press Ctrl+C in the terminal to stop.

## Privacy and security

- The server binds to `127.0.0.1` by default and refuses other interfaces
  unless you pass `--allow-remote` (which exposes your images to the network).
  With a wildcard bind such as `--host 0.0.0.0`, requests for any host name are
  accepted, so the session token is then the only protection; traffic is plain
  HTTP, so use this only on a network you trust.
- Every API call needs the random per-session token in the printed address.
  Requests from other websites are rejected by Host and Origin checks, and the
  page loads no remote scripts, fonts or images, so pixel data never leaves
  your computer.
- Uploaded files are copied to a private temporary folder and deleted when the
  studio stops. Files opened by path are read in place.

## Workflow

**1. Load data.** Use the *Data* tab, or pass it on the command line:

| Source | How |
| --- | --- |
| NIfTI (`.nii`, `.nii.gz`) | Path or upload. A registered label map can be added as the mask; tick *Resample mask to image grid* for registered masks on a different grid. |
| NumPy (`.npy`) | Path or upload. Arrays use unit spacing and unknown patient geometry. |
| DICOM | A series folder or single file, by path. *Strict DICOM* matches `load_dicom_series(strict=True)`. |
| Phantoms | MRI, CT, X-ray (2D) and DBT (3D). *Demo labels & targets* adds a synthetic two-class mask, two landmarks and a box so every overlay can be exercised. |
| Extra channels | Name a registered image (for example `flair`) and add it by path or upload. NIfTI channels must share the primary image's patient grid. |

The *Summary* card lists shape, spacing, geometry, intensity statistics, label
volumes in mm³ (or mm² in 2D) and any validator findings on the input itself.

**2. Build the policy.** Start from a modality preset or press *+ Add step* and
search the transform palette. Select a step to edit it in the inspector on the
right:

- Fields adapt to each value: sliders for `p`, paired inputs for ranges,
  dropdowns for documented choices, and checkboxes for flags. The `{ }` button
  switches any field to raw JSON, for example to turn a scalar into a range.
- Each parameter shows its documented meaning and default; *reset* restores
  the default.
- *Model card* shows the transform's assumptions, units, limitations and
  references.
- Steps can be moved, duplicated, deleted, or wrapped in `Compose`, `OneOf`,
  `SomeOf`, `Guard` or `PerChannel`. *Unwrap* reverses a wrap.
- Undo and redo cover every edit (Ctrl/Cmd+Z, Shift for redo).

A policy that cannot be built is reported in the status chip, and the failing
step is highlighted with the constructor's error message.

**3. Preview.** The *Preview* tab shows the input and augmented slices side by
side, with masks, landmarks and boxes drawn over them:

- *Seed*, *Sample ID*, *Epoch* and *Draw* use exactly the per-sample seeding of
  `apply_for_sample(policy, volume, seed=…, sample_id=…, epoch=…,
  augmentation_index=…)`, so you can reproduce what training saw for any sample.
  A sample ID made only of digits is sent as an integer.
- Choose the slice axis and position, the displayed channel, the mask opacity
  and the display window: shared with the input (intensity changes stay
  visible), automatic per image, or manual.
- *Difference* adds an absolute-difference view when the grid is unchanged.
- Scroll to zoom and drag to pan; both views move together. Double-click resets.
- *Steps this draw* lists every step as applied or skipped, with timings. Click
  a row to select that step. *Validation* lists `VolumeValidator` findings for
  the draw against the input.
- *Download replay record* saves a `record_augmentation` record for the draw;
  `replay_augmentation(record, volume)` reproduces it exactly in the same
  environment.

Keyboard: ←/→ change slice, N/P change draw, R picks a random seed.

**4. Survey variability.** The *Gallery* renders consecutive draw indices at the
current slice, each tagged with its validation status and the steps that ran.
Click a tile to open that draw in the preview.

**5. Stress-test.** The *Stress test* runs up to 500 draws (stopping after two
minutes) and validates each against the input, using validator settings you
can edit. It reports:

- pass, rejection and exception rates;
- validation findings by check;
- exception messages;
- output shapes;
- mean and 95th-percentile time per draw;
- how often each step actually ran, next to its configured `p`. Children of
  `OneOf`/`SomeOf` run less often than their own `p`, which this makes visible.

**6. Export.** *Export* produces:

- **JSON** — the versioned, hash-checked document from `to_json`; load it with
  `from_json`. Prefer this for experiments.
- **YAML** — the same document via `to_yaml` (requires PyYAML).
- **Python** — readable constructor code that omits default values.

*Import* accepts any JSON or YAML policy produced by the library or the studio.

## From Python and notebooks

```python
from medaugmentx.io import load_nifti
from medaugmentx.presets import mri_pipeline
from medaugmentx.studio import launch

volume = load_nifti("t1.nii.gz", mask_path="seg.nii.gz")
studio = launch(mri_pipeline(), volume, block=False)  # opens the browser
# ... tune the policy in the studio ...
policy = studio.current_policy()  # a ready-to-use Transform
studio.stop()
```

`launch(..., block=True)` (the default) serves until interrupted, like the
command-line tool. `StudioServer` and `StudioSession` are available for custom
embedding and automated tests.

## Command-line reference

```text
medaugmentx studio [--image PATH | --phantom {mri,ct,xray,dbt}] [--mask PATH]
                   [--resample-mask] [--policy FILE] [--host HOST] [--port PORT]
                   [--token TOKEN] [--allow-remote] [--no-browser] [--verbose]
```

`--port 0` (the default) picks a free port. `--token` fixes the session token,
which is useful for scripted access. `--verbose` logs every request.

## Limits

- One volume is loaded at a time. Each preview, gallery tile and stress draw
  runs the full policy, so large volumes with heavy steps (elastic deformation,
  3D resampling) take correspondingly longer. Changing only the slice, window
  or overlay reuses the last draw.
- Custom transforms appear once they are registered with `register_transform`
  in the Python process that runs the studio, for example through `launch()`.
- The studio previews structural effects; like every validator finding, it is
  not a clinical plausibility assessment.
