"""All MedAugmentX transforms re-exported as a flat namespace."""
# Spatial
# Intensity
from medaugmentx.transforms.intensity.bias_field import BiasField
from medaugmentx.transforms.intensity.blur import GaussianBlur, MedianBlur, SimulateLowResolution
from medaugmentx.transforms.intensity.brightness_contrast import BrightnessContrast
from medaugmentx.transforms.intensity.clahe import CLAHEContrast
from medaugmentx.transforms.intensity.contrast import GammaCorrection
from medaugmentx.transforms.intensity.histogram import HistogramMatch
from medaugmentx.transforms.intensity.noise import GaussianNoise, RicianNoise
from medaugmentx.transforms.intensity.sharpen import Sharpen
from medaugmentx.transforms.intensity.window_level import WindowLevel

# CT
from medaugmentx.transforms.modality.ct import BeamHardening, MetalStreak

# MRI
from medaugmentx.transforms.modality.mri import GhostingArtifact, KSpaceDropout, MRIMotion

# Tomosynthesis (DBT)
from medaugmentx.transforms.modality.tomosynthesis import (
    AnisotropicElastic,
    CompressionVariation,
    LimitedAngleBlur,
    ReconStreak,
    SlabShift,
    SliceDropout,
)

# X-ray (DXR)
from medaugmentx.transforms.modality.xray import GridArtifact, ScatterSimulation
from medaugmentx.transforms.spatial.affine import RandomAffine
from medaugmentx.transforms.spatial.crop import AnatomicCrop
from medaugmentx.transforms.spatial.dropout import CoarseDropout
from medaugmentx.transforms.spatial.elastic import ElasticDeform
from medaugmentx.transforms.spatial.flip import RandomFlip
from medaugmentx.transforms.spatial.resize import CenterCrop, Pad, Resize

__all__ = [
    # Spatial
    "RandomAffine",
    "RandomFlip",
    "AnatomicCrop",
    "ElasticDeform",
    "CoarseDropout",
    "Resize",
    "Pad",
    "CenterCrop",
    # Intensity
    "RicianNoise",
    "GaussianNoise",
    "GammaCorrection",
    "BiasField",
    "WindowLevel",
    "BrightnessContrast",
    "GaussianBlur",
    "MedianBlur",
    "SimulateLowResolution",
    "Sharpen",
    "CLAHEContrast",
    "HistogramMatch",
    # Tomosynthesis (DBT)
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
    # X-ray (DXR)
    "ScatterSimulation",
    "GridArtifact",
]
