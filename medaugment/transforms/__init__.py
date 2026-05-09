"""All MedAugment transforms re-exported as a flat namespace."""
# Spatial
from medaugment.transforms.spatial.affine import RandomAffine
from medaugment.transforms.spatial.crop import AnatomicCrop
from medaugment.transforms.spatial.elastic import ElasticDeform
from medaugment.transforms.spatial.flip import RandomFlip

# Intensity
from medaugment.transforms.intensity.bias_field import BiasField
from medaugment.transforms.intensity.blur import GaussianBlur, SimulateLowResolution
from medaugment.transforms.intensity.brightness_contrast import BrightnessContrast
from medaugment.transforms.intensity.contrast import GammaCorrection
from medaugment.transforms.intensity.noise import GaussianNoise, RicianNoise
from medaugment.transforms.intensity.window_level import WindowLevel

# Tomosynthesis (DBT)
from medaugment.transforms.modality.tomosynthesis import (
    AnisotropicElastic,
    LimitedAngleBlur,
    SlabShift,
    SliceDropout,
)

# MRI
from medaugment.transforms.modality.mri import GhostingArtifact, KSpaceDropout

# CT
from medaugment.transforms.modality.ct import BeamHardening

__all__ = [
    # Spatial
    "RandomAffine",
    "RandomFlip",
    "AnatomicCrop",
    "ElasticDeform",
    # Intensity
    "RicianNoise",
    "GaussianNoise",
    "GammaCorrection",
    "BiasField",
    "WindowLevel",
    "BrightnessContrast",
    "GaussianBlur",
    "SimulateLowResolution",
    # Tomosynthesis (DBT)
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
