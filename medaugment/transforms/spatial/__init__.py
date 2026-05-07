"""Spatial transforms: affine, flip, crop, elastic deform."""
from medaugment.transforms.spatial.affine import RandomAffine
from medaugment.transforms.spatial.crop import AnatomicCrop
from medaugment.transforms.spatial.elastic import ElasticDeform
from medaugment.transforms.spatial.flip import RandomFlip

__all__ = ["RandomAffine", "RandomFlip", "AnatomicCrop", "ElasticDeform"]
