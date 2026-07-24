"""MedAugmentX — clinically-aware medical image augmentation.

Public surface for Phase 3.
"""
from medaugmentx.core import (
    Compose,
    MedVolume,
    OneOf,
    SomeOf,
    Transform,
)
from medaugmentx.inspection import PipelineStep, iter_pipeline, pipeline_summary
from medaugmentx.validation import (
    Guard,
    ValidationError,
    ValidationIssue,
    ValidationReport,
    VolumeValidator,
)

__version__ = "0.9.0"

__all__ = [
    "__version__",
    "MedVolume",
    "Transform",
    "Compose",
    "OneOf",
    "SomeOf",
    "PipelineStep",
    "iter_pipeline",
    "pipeline_summary",
    "Guard",
    "VolumeValidator",
    "ValidationReport",
    "ValidationIssue",
    "ValidationError",
]
