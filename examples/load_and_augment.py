"""Load a NIfTI / DICOM volume, augment, save back.

Skip the loaders gracefully if the optional backend is not installed.
"""
from __future__ import annotations

import argparse
import sys

from medaugment import Compose
from medaugment.transforms import (
    ElasticDeform,
    GammaCorrection,
    GaussianNoise,
    RandomAffine,
    RandomFlip,
)


def build_pipeline(seed: int) -> Compose:
    return Compose([
        RandomFlip(axes=("x",), p_per_axis=0.5),
        RandomAffine(rotation=10.0, scale=(0.95, 1.05), p=0.7),
        ElasticDeform(alpha=20.0, sigma=4.0, p=0.5),
        GaussianNoise(std=0.01, p=0.5),
        GammaCorrection(gamma=(0.9, 1.1), p=0.4),
    ], seed=seed)


def load(path: str):
    if path.endswith(".nii") or path.endswith(".nii.gz"):
        from medaugment.io import load_nifti

        return "nifti", load_nifti(path)
    # otherwise treat as DICOM directory
    from medaugment.io import load_dicom_series

    return "dicom", load_dicom_series(path)


def save_if_possible(kind: str, vol, path: str | None) -> None:
    if path is None:
        return
    if kind == "nifti":
        from medaugment.io import save_nifti

        save_nifti(vol, path)
        print(f"saved -> {path}")
    else:
        print("DICOM writing is out of scope for Phase 1; skipped save.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Augment a NIfTI file or DICOM directory.")
    parser.add_argument("input", help="Path to .nii/.nii.gz file or DICOM directory")
    parser.add_argument("--out", help="Where to save the augmented volume (NIfTI only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        kind, vol = load(args.input)
    except ImportError as exc:
        print(f"Missing optional dependency: {exc}", file=sys.stderr)
        return 2

    print("loaded :", vol)
    pipeline = build_pipeline(seed=args.seed)
    out = pipeline(vol)
    print("output :", out)

    save_if_possible(kind, out, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
