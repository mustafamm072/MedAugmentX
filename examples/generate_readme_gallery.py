"""Render the README gallery from synthetic phantoms and actual library outputs.

Run from the repository root after installing medaugmentx and matplotlib:
    python examples/generate_readme_gallery.py

No patient data, downloads, or image-generation services are used. Transform
strengths are chosen for illustration, not as recommended training policies.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from medaugmentx.phantoms import ct_phantom, dbt_phantom, mri_phantom, xray_phantom
from medaugmentx.transforms import (
    BeamHardening,
    BiasField,
    GridArtifact,
    LimitedAngleBlur,
    MetalStreak,
    ReconStreak,
    RicianNoise,
    ScatterSimulation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docs/assets"
    )
    args = parser.parse_args()
    # Keep font caches outside the checkout, and support headless regeneration.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "medaugmentx-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [
        (
            "MRI",
            "Synthetic axial slice",
            mri_phantom(),
            [BiasField(alpha=0.65, coarse_shape=3, seed=7), RicianNoise(std=0.09, seed=7)],
            (0, 1),
        ),
        (
            "CT",
            "Synthetic axial slice",
            ct_phantom(),
            [BeamHardening(alpha=0.10, seed=7), MetalStreak(intensity=0.16, num_streaks=9, seed=7)],
            (-250, 350),
        ),
        (
            "X-RAY",
            "Synthetic projection",
            xray_phantom(),
            [
                ScatterSimulation(fraction=0.5, sigma=22, seed=7),
                GridArtifact(amplitude=0.16, frequency=0.16, seed=7),
            ],
            (0, 0.85),
        ),
        (
            "TOMOSYNTHESIS",
            "Synthetic 3D volume / depth view",
            dbt_phantom(),
            [
                LimitedAngleBlur(arc_degrees=15, base_sigma=1.8, seed=7),
                ReconStreak(amplitude=0.3, num_planes=3, displacement=4, seed=7),
            ],
            (0, 0.8),
        ),
    ]
    fig = plt.figure(figsize=(12, 15.4), facecolor="#f4f6f8")
    fig.text(0.055, 0.972, "See what changes.", fontsize=28, weight="bold", color="#152b3c")
    fig.text(
        0.055,
        0.946,
        "Four modalities. Real MedAugmentX transforms. Reproducible synthetic inputs.",
        fontsize=12,
        color="#52616e",
    )
    records = []
    for row, (name, subtitle, volume, transforms, limits) in enumerate(cases):
        top = 0.914 - row * 0.215
        fig.text(0.055, top, name, fontsize=11, weight="bold", color="#087e8b")
        fig.text(0.245, top, subtitle, fontsize=10, color="#52616e")
        outputs = [volume, *(transform(volume) for transform in transforms)]
        titles = ["Original", *(type(transform).__name__ for transform in transforms)]
        for column, (output, title) in enumerate(zip(outputs, titles)):
            assert np.isfinite(output.image).all()
            assert output.shape == volume.shape
            if column:
                assert not np.array_equal(output.image, volume.image)
            plane = output.image if output.ndim == 2 else output.image[:, output.shape[1] // 2, :]
            ax = fig.add_axes((0.055 + column * 0.31, top - 0.189, 0.27, 0.168))
            extent = (0, 64, 64, 0) if output.ndim == 3 else None
            # Same display range for all three panels; no per-image normalization.
            ax.imshow(
                plane,
                cmap="gray",
                vmin=limits[0],
                vmax=limits[1],
                interpolation="nearest",
                extent=extent,
            )
            ax.set_title(title, fontsize=11, color="#152b3c", pad=7)
            ax.set_axis_off()
        records.append(
            {
                "modality": name,
                "display_limits": limits,
                "shape": volume.shape,
                "spacing": volume.spacing,
                "view": subtitle,
                "transforms": [t.to_dict() for t in transforms],
            }
        )
    fig.text(
        0.055,
        0.028,
        "Synthetic illustrations, not patient scans. Effects are emphasized for visibility.",
        fontsize=10,
        color="#52616e",
    )
    fig.text(
        0.055,
        0.013,
        "Each transform is applied independently to the original. Grayscale limits are fixed within each row.",
        fontsize=9,
        color="#52616e",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "modality-gallery.png"
    fig.savefig(
        path, dpi=140, facecolor=fig.get_facecolor(), metadata={"Software": "MedAugmentX gallery"}
    )
    plt.close(fig)
    manifest = {
        "synthetic": True,
        "environment": {
            name: version(name) for name in ("medaugmentx", "numpy", "scipy", "matplotlib")
        },
        "examples": records,
    }
    (args.output_dir / "modality-gallery.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
