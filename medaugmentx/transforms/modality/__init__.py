"""Modality-specific augmentation modules."""
from medaugmentx.transforms.modality.ct import BeamHardening, MetalStreak
from medaugmentx.transforms.modality.mri import GhostingArtifact, KSpaceDropout, MRIMotion
from medaugmentx.transforms.modality.tomosynthesis import (
    AnisotropicElastic,
    CompressionVariation,
    LimitedAngleBlur,
    ReconStreak,
    SlabShift,
    SliceDropout,
)
from medaugmentx.transforms.modality.xray import GridArtifact, ScatterSimulation

__all__ = [
    # Tomosynthesis / DBT
    "SlabShift",
    "LimitedAngleBlur",
    "SliceDropout",
    "AnisotropicElastic",
    "CompressionVariation",
    "ReconStreak",
    # MRI
    "GhostingArtifact",
    "KSpaceDropout",
    "MRIMotion",
    # CT
    "BeamHardening",
    "MetalStreak",
    # X-ray / DXR
    "ScatterSimulation",
    "GridArtifact",
]
