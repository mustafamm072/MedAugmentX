# Benchmarks

A lightweight, dependency-free speed benchmark for MedAugmentX transforms.
It times every registered leaf transform on a synthetic volume and prints a
table sorted by wall-clock time.

```bash
# Typical training patch size (default)
python benchmarks/benchmark.py

# Phase-3 acceptance-target size (512x512x80 DBT volume)
python benchmarks/benchmark.py --shape 80 512 512

# More repeats / a 2-D volume
python benchmarks/benchmark.py --shape 512 512 --repeats 5
```

## Interpreting the results

- Times are **best-of-N** wall-clock per `__call__`, in milliseconds, on a
  single CPU thread (whatever BLAS/SciPy threading is configured).
- Container transforms (`Compose`, `OneOf`, `SomeOf`) and `HistogramMatch`
  (which needs a reference array) are skipped.
- 3-D-only transforms (`SlabShift`, `ReconStreak`, …) are skipped for 2-D
  shapes.

## On the 500 ms CPU target

The Phase-3 / v1.0 acceptance criterion is *"all transforms complete in
< 500 ms on CPU for a 512×512×80 DBT volume."* On a full-resolution
21-megavoxel volume this is **not yet met on CPU** — and not only by the new
transforms: several Phase-1/2 transforms (`ElasticDeform`, `BiasField`,
`AnisotropicElastic`, `RandomAffine`, `SimulateLowResolution`) also exceed it.
Closing that gap is the job of the planned **PyTorch GPU backend**
(roadmap item 3.4), which targets a ≥ 5× speedup for spatial transforms.

At typical training patch sizes (e.g. `64×256×256`) the large majority of
transforms run in well under 50 ms. The intrinsically expensive ones are
3-D `MedianBlur` (a cubic-window rank filter) and large-sigma Gaussian
operations (`ScatterSimulation`, `BiasField`); budget accordingly or run them
at lower probability.
