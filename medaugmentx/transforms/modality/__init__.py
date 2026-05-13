"""Modality-specific augmentation modules."""
from medaugmentx.transforms.modality.ct import BeamHardening
from medaugmentx.transforms.modality.mri import GhostingArtifact, KSpaceDropout
from medaugmentx.transforms.modality.tomosynthesis import (
    AnisotropicElastic,
    LimitedAngleBlur,
    SlabShift,
    SliceDropout,
)

__all__ = [
    # Tomosynthesis / DBT
    "SlabShift",
    "LimitedAngleBlur",
    "SliceDropout",
    "AnisotropicElastic",
    # MRI
    "GhostingArtifact",
    "KSpaceDropout",
    # CT
    "BeamHardening",
]
