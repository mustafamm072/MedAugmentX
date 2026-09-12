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

The pipeline is seeded, so the same seed reproduces the same draw sequence on
any machine, and it round-trips through JSON or YAML for an experiment record.

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

**Modality physics, not generic image ops.** MRI k-space dropout, CT beam
hardening, X-ray scatter, and limited-angle tomosynthesis blur are modelled as
the acquisition artifacts they are, not approximated with a blur kernel. The
:doc:`catalog <catalog>` lists what each one simulates.

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

.. toctree::
   :caption: Guides
   :hidden:

   Research & reproducibility <../RESEARCH_GUIDE>
   Commercial adoption <../COMMERCIAL_ADOPTION>
   Benchmarks <../../benchmarks/README>

.. toctree::
   :caption: Project
   :hidden:

   Roadmap <../MILESTONES>
   Contributing <../../CONTRIBUTING>
   Working on the docs <../README>
   Security policy <../../SECURITY>
   Changelog <../../CHANGELOG>
