"""MRI-specific augmentation transforms."""
from medaugment.transforms.modality.mri.ghosting import GhostingArtifact
from medaugment.transforms.modality.mri.kspace import KSpaceDropout

__all__ = ["GhostingArtifact", "KSpaceDropout"]
