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

from medaugmentx.presets import ct_pipeline, mri_pipeline


def build_mri_pipeline(seed: int):
    return mri_pipeline(seed=seed, normalize=True)


def build_ct_pipeline(seed: int):
    return ct_pipeline(seed=seed)


def load(path: str, mask_path: str | None = None):
    if path.endswith(".nii") or path.endswith(".nii.gz"):
        from medaugmentx.io import load_nifti

        return "nifti", load_nifti(path, mask_path=mask_path)
    if mask_path is not None:
        raise ValueError("--mask currently supports registered NIfTI inputs only")
    from medaugmentx.io import load_dicom_series

    return "dicom", load_dicom_series(path)


def save_if_possible(kind: str, vol, path: str | None, mask_path: str | None = None) -> None:
    if path is None:
        return
    if kind == "nifti":
        from medaugmentx.io import save_nifti

        save_nifti(vol, path, mask_path=mask_path, allow_assumed_geometry=False)
        print(f"saved -> {path}")
    else:
        print("DICOM writing is not supported; use --out with a NIfTI input to save.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Augment a NIfTI file or DICOM directory.")
    parser.add_argument("input", help="Path to .nii/.nii.gz file or DICOM directory")
    parser.add_argument("--out", help="Where to save the augmented volume (NIfTI only)")
    parser.add_argument("--mask", help="Registered NIfTI label map")
    parser.add_argument("--mask-out", help="Augmented label-map output (requires --mask and --out)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preset",
        choices=["mri", "ct"],
        default="mri",
        help="Which built-in pipeline to use (default: mri)",
    )
    args = parser.parse_args()

    if args.mask_out and (not args.mask or not args.out):
        parser.error("--mask-out requires --mask and --out")
    try:
        kind, vol = load(args.input, args.mask)
    except ImportError as exc:
        print(f"Missing optional dependency: {exc}", file=sys.stderr)
        print('Install with: pip install "medaugmentx[io]"', file=sys.stderr)
        return 2

    print("loaded :", vol)

    if args.preset == "ct":
        pipeline = build_ct_pipeline(seed=args.seed)
    else:
        pipeline = build_mri_pipeline(seed=args.seed)

    out = pipeline(vol)
    print("output :", out)

    save_if_possible(kind, out, args.out, args.mask_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
