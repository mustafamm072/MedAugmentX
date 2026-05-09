"""MRI k-space corruption (line dropout and spike artifacts)."""
from __future__ import annotations

from typing import Any, Union

import numpy as np

from medaugment.core.base import Transform
from medaugment.core.utils import SeedLike, as_float32
from medaugment.core.volume import MedVolume

Range = Union[float, tuple[float, float]]


class KSpaceDropout(Transform):
    """Zero a fraction of k-space lines to simulate acquisition errors.

    Real MRI scanners occasionally produce corrupted k-space lines from
    hardware glitches, RF interference, or momentary gradient failure.
    This transform simulates that by:

    1. Taking the 2-D FFT of each slice (or the 2-D image).
    2. Randomly zeroing ``dropout_fraction`` of lines along the phase-encode
       direction.
    3. Reconstructing the magnitude image via inverse FFT.

    The result exhibits Gibbs ringing / truncation artifacts around edges —
    the correct physics for this failure mode.

    Works on 2-D and 3-D volumes.  For 3-D, each axial slice is corrupted
    independently (independent random line selection per slice).

    Args:
        dropout_fraction: Fraction of k-space lines to zero.  Scalar → fixed;
            tuple → sampled from ``(lo, hi)``.
        phase_encode_axis: Axis along which lines are dropped (``"y"`` rows or
            ``"x"`` columns).
        p: Probability of applying.
        seed: RNG seed.
    """

    def __init__(
        self,
        dropout_fraction: Range = (0.01, 0.05),
        phase_encode_axis: str = "y",
        p: float = 1.0,
        seed: SeedLike = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if isinstance(dropout_fraction, (int, float)):
            v = float(dropout_fraction)
            if not 0 <= v <= 1:
                raise ValueError("dropout_fraction must be in [0, 1]")
            self.dropout_range: tuple[float, float] = (v, v)
        else:
            lo, hi = float(dropout_fraction[0]), float(dropout_fraction[1])
            if lo < 0 or hi < lo or hi > 1.0:
                raise ValueError(f"dropout_fraction range invalid: {dropout_fraction}")
            self.dropout_range = (lo, hi)
        if phase_encode_axis not in ("x", "y"):
            raise ValueError("phase_encode_axis must be 'x' or 'y'")
        self.phase_encode_axis = phase_encode_axis

    def _corrupt_slice(self, img2d: np.ndarray, frac: float) -> np.ndarray:
        """Corrupt a single 2-D plane in k-space."""
        pe_axis = 0 if self.phase_encode_axis == "y" else 1
        n_lines = img2d.shape[pe_axis]
        n_drop = max(1, int(round(frac * n_lines)))

        kspace = np.fft.fft2(img2d.astype(np.complex64))
        drop_idxs = self.rng.choice(n_lines, size=n_drop, replace=False)
        if pe_axis == 0:
            kspace[drop_idxs, :] = 0.0
        else:
            kspace[:, drop_idxs] = 0.0

        return np.abs(np.fft.ifft2(kspace)).astype(np.float32)

    def apply(self, volume: MedVolume) -> MedVolume:
        image = as_float32(volume.image)
        frac = float(self.rng.uniform(*self.dropout_range))

        if frac == 0.0:
            return volume

        if volume.is_3d:
            out = np.stack(
                [self._corrupt_slice(image[z], frac) for z in range(image.shape[0])],
                axis=0,
            )
        else:
            out = self._corrupt_slice(image, frac)

        return volume.replace(image=out)

    def to_dict(self) -> dict[str, Any]:
        dr = self.dropout_range
        return {
            "name": self.__class__.__name__,
            "params": {
                "dropout_fraction": dr[0] if dr[0] == dr[1] else list(dr),
                "phase_encode_axis": self.phase_encode_axis,
                "p": self.p,
            },
        }
