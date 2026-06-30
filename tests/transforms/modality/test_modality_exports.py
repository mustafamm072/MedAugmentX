from medaugmentx.transforms.modality import (
    CompressionVariation,
    GridArtifact,
    MetalStreak,
    MRIMotion,
    ReconStreak,
    ScatterSimulation,
)


def test_new_modality_transforms_exported_from_root_package():
    assert CompressionVariation.__name__ == "CompressionVariation"
    assert ReconStreak.__name__ == "ReconStreak"
    assert MRIMotion.__name__ == "MRIMotion"
    assert MetalStreak.__name__ == "MetalStreak"
    assert ScatterSimulation.__name__ == "ScatterSimulation"
    assert GridArtifact.__name__ == "GridArtifact"
