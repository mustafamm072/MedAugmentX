MedAugmentX
===========

Clinically-aware augmentation for medical image AI training, on NumPy and SciPy
alone. Every transform operates on a :class:`~medaugmentx.MedVolume` — an image
plus its mask, voxel spacing, metadata, and optional keypoints and boxes — and
returns a new one, so annotations stay consistent with the pixels.

.. code-block:: bash

   pip install medaugmentx

.. code-block:: python

   from medaugmentx import Compose, MedVolume
   from medaugmentx.transforms import BiasField, RandomAffine, RicianNoise

   pipeline = Compose([
       RandomAffine(rotation=10, scale=(0.95, 1.05)),
       BiasField(alpha=0.3),
       RicianNoise(std=0.02, p=0.5),
   ], seed=0)

   augmented = pipeline(MedVolume(image, mask=mask, spacing=(1.0, 0.9, 0.9)))

Fresh equivalent pipelines with the same integer seed reproduce the draw
sequence in the same software environment. JSON or YAML stores configuration,
not an in-progress random-state checkpoint.

Where to start
--------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - If you want to
     - Read

   * - Understand what the library does and install it
     - :doc:`Project overview <../../README>`

   * - Pick the right transform for a modality
     - :doc:`Transform catalog <catalog>`

   * - Copy a working snippet for your framework
     - :doc:`Recipes <../API_EXAMPLES>`

   * - Look up an exact signature or return type
     - :doc:`API reference <api/index>`

   * - Work through a full MRI, CT, or DBT example
     - :doc:`Tutorial notebooks <../../notebooks/README>`

   * - Make results reproducible in a paper
     - :doc:`Research guide <../RESEARCH_GUIDE>`

   * - Take it to production
     - :doc:`Commercial adoption <../COMMERCIAL_ADOPTION>`

   * - Write your own transform
     - :doc:`Architecture <../ARCHITECTURE>`

What makes it different
-----------------------

**Modality-specific approximations.** MRI k-space dropout, CT cupping,
X-ray scatter, and limited-angle tomosynthesis blur offer controllable
appearance perturbations. Several use simple image-domain kernels; these
are not validated scanner simulators. The :doc:`catalog <catalog>` lists
the implemented effects.

**Annotations move with the pixels.** Spatial transforms map masks, keypoints,
and bounding boxes through the same sampled geometry, using nearest-neighbour
interpolation for labels. The catalog's *Targets* column says which transforms
warp targets and which leave them alone.

**Two dependencies.** NumPy and SciPy. DICOM, NIfTI, PyTorch, MONAI, and TorchIO
support are optional extras that are imported only when used.

**Reproducible by construction.** Every transform takes a ``seed``; containers
derive an independent stream per child rather than sharing one. Pipelines
serialise to JSON or YAML and reload identically.

**Validated output.** :class:`~medaugmentx.VolumeValidator` and
:class:`~medaugmentx.Guard` check that an augmented volume is still plausible —
finite, non-collapsed, labels intact — and can raise, warn, revert, or retry.

.. toctree::
   :caption: Getting started
   :hidden:

   Overview <../../README>
   Recipes <../API_EXAMPLES>
   Tutorial notebooks <../../notebooks/README>
   Example scripts <../../examples/README>

.. toctree::
   :caption: Reference
   :hidden:

   Transform catalog <catalog>
   api/index
   Public surface & semantics <../API_REFERENCE>
   Architecture <../ARCHITECTURE>
   Model cards <../MODEL_CARDS>

.. toctree::
   :caption: Guides
   :hidden:

   Version 1 migration <../MIGRATION_V1>
   Research & reproducibility <../RESEARCH_GUIDE>
   Commercial adoption <../COMMERCIAL_ADOPTION>
   Benchmarks <../../benchmarks/README>

.. toctree::
   :caption: Project
   :hidden:

   Roadmap <../MILESTONES>
   Project review <../PROJECT_REVIEW>
   Release readiness <../RELEASE_READINESS>
   Version 1.0.0 release notes <../RELEASE_NOTES_V1>
   Contributing <../../CONTRIBUTING>
   Working on the docs <../README>
   Security policy <../../SECURITY>
   Changelog <../../CHANGELOG>
