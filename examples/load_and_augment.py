"""Load a NIfTI / DICOM volume, augment, save back.

Skip the loaders gracefully if the optional backend is not installed.

Run with:

    # NIfTI
    python examples/load_and_augment.py brain_t1.nii.gz --out brain_augmented.nii.gz

    # DICOM directory (save skipped — no DICOM writer)
    python examples/load_and_augment.py /path/to/CT_series/

    # Use the built-in MRI preset
    python examples/load_and_augment.py brain_t1.nii.gz --preset mri --out out.nii.gz
"""
from __future__ import annotations

import argparse
import sys

from medaugmentx import Compose
from medaugmentx.transforms import (
    BiasField,
    ElasticDeform,
    GammaCorrection,
    GaussianNoise,
    RandomAffine,
    RandomFlip,
    WindowLevel,
)


def build_mri_pipeline(seed: int) -> Compose:
    from medaugmentx.transforms import RicianNoise
    return Compose([
        RandomFlip(axes=("x",), p_per_axis=0.5),
        RandomAffine(rotation=10.0, scale=(0.9, 1.1), translation=(-0.05, 0.05), p=0.7),
        ElasticDeform(alpha=30.0, sigma=4.0, p=0.5),
        BiasField(alpha=0.3, p=0.7),
        RicianNoise(std=(0.005, 0.02), p=0.5),
        GammaCorrection(gamma=(0.85, 1.15), p=0.5),
    ], seed=seed)


def build_ct_pipeline(seed: int) -> Compose:
    return Compose([
        RandomFlip(axes=("x",), p_per_axis=0.5),
        RandomAffine(rotation=8.0, scale=(0.9, 1.1), translation=(-0.03, 0.03), p=0.7),
        ElasticDeform(alpha=20.0, sigma=4.0, p=0.4),
        WindowLevel(center_shift_frac=0.05, width_scale=(0.85, 1.15), p=0.6),
        GaussianNoise(std=(5.0, 20.0), p=0.4),
        GammaCorrection(gamma=(0.9, 1.1), p=0.4),
    ], seed=seed)


def load(path: str):
    if path.endswith(".nii") or path.endswith(".nii.gz"):
        from medaugmentx.io import load_nifti
        return "nifti", load_nifti(path)
    from medaugmentx.io import load_dicom_series
    return "dicom", load_dicom_series(path)


def save_if_possible(kind: str, vol, path: str | None) -> None:
    if path is None:
        return
    if kind == "nifti":
        from medaugmentx.io import save_nifti
        save_nifti(vol, path)
        print(f"saved -> {path}")
    else:
        print("DICOM writing is not supported; use --out with a NIfTI input to save.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Augment a NIfTI file or DICOM directory.")
    parser.add_argument("input", help="Path to .nii/.nii.gz file or DICOM directory")
    parser.add_argument("--out", help="Where to save the augmented volume (NIfTI only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preset",
        choices=["mri", "ct"],
        default="mri",
        help="Which built-in pipeline to use (default: mri)",
    )
    args = parser.parse_args()

    try:
        kind, vol = load(args.input)
    except ImportError as exc:
        print(f"Missing optional dependency: {exc}", file=sys.stderr)
        print("Install with: pip install \"medaugmentx[io]\"", file=sys.stderr)
        return 2

    print("loaded :", vol)

    if args.preset == "ct":
        pipeline = build_ct_pipeline(seed=args.seed)
    else:
        pipeline = build_mri_pipeline(seed=args.seed)

    out = pipeline(vol)
    print("output :", out)

    save_if_possible(kind, out, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
