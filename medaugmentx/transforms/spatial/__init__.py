"""Spatial transforms: affine, flip, crop, elastic deform."""
from medaugmentx.transforms.spatial.affine import RandomAffine
from medaugmentx.transforms.spatial.crop import AnatomicCrop
from medaugmentx.transforms.spatial.elastic import ElasticDeform
from medaugmentx.transforms.spatial.flip import RandomFlip

__all__ = ["RandomAffine", "RandomFlip", "AnatomicCrop", "ElasticDeform"]
