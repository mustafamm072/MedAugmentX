# Benchmarks and evaluation

Run from an installed checkout:

```bash
python benchmarks/benchmark.py --shape 32 128 128 --repeats 10 --threads 1 --json benchmark.json
python benchmarks/benchmark.py --shape 80 512 512 --repeats 5 --json full-volume.json
```

Each transform runs in a fresh process. The script sets common BLAS/OpenMP thread
environment variables before the child imports NumPy, warms up once, and records
all wall-clock samples, median and p95. Short runs have noisy p95 estimates.
JSON records library/Python/NumPy/SciPy versions, platform, CPU description,
logical CPU count, thread environment, NumPy build configuration, input and mask
dtypes, shapes, seed and complete constructor policy. Containers are excluded;
histogram matching uses a fixed quantile reference. 3D-only effects skip 2D input.

Memory is measured in an extra draw to avoid tracing overhead in timings.
`tracked_peak_bytes` is tracemalloc's observed allocation peak, which may miss
native allocations. `process_peak_rss_bytes` is process lifetime peak RSS,
including interpreter/imports/input/warm-up/timing; it is not an incremental
transform cost and is unavailable on Windows. Compare identical environments
and settings. Small shapes do not establish full-resolution performance.

`resources/validation/benchmark-small-3d.json` and `benchmark-patch-3d.json`
(32×128×128) are measured smoke/patch benchmarks,
not a performance guarantee or comparison against other libraries. There is no
universal 500 ms claim and no GPU backend in v1. Budget using your own volumes,
annotation load and policy. Avoid large 3D median kernels or unnecessary full
volume augmentation when patch-based training suffices.

## Controlled synthetic segmentation comparison

```bash
pip install 'medaugmentx[torch]'
python benchmarks/segmentation_experiment.py --json experiment.json --epochs 12 --seeds 0 1 2
```

The script uses fixed disjoint synthetic ellipse-image splits, a small CPU CNN,
identical training settings, no augmentation/spatial/appearance ablations and
three initialization seeds. The held-out split has an intensity/noise shift.
JSON contains policies, environments, per-image Dice and training runtime.
No policy or threshold is selected using the held-out split.

The checked-in run produced mean Dice **0.9091** without augmentation,
**0.9062** with spatial augmentation and **0.8596** with the tested appearance
policy (seed standard deviations 0.0096, 0.0162, 0.0022). Appearance augmentation
hurt this task. This small synthetic comparison demonstrates an evaluation
workflow and does not establish medical dataset benefit. It is not evidence to
recommend one preset for clinical data. The raw run is in
`resources/validation/synthetic-segmentation.json`.

For a deployment decision, repeat a preregistered ablation on patient-separated
training/validation/test cohorts, keep evaluation preprocessing fixed, assess
rare classes and scanner/site shifts, and report uncertainty and Guard fallback
rates. Real scanner/cohort evaluation has not been performed in this repository.
