"""Modality-specific augmentation modules."""
from medaugment.transforms.modality.ct import BeamHardening
from medaugment.transforms.modality.mri import GhostingArtifact, KSpaceDropout
from medaugment.transforms.modality.tomosynthesis import (
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
