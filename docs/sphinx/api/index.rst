API reference
=============

Signatures and prose on these pages are generated from the source at build time,
so they always describe the version of MedAugmentX you are reading the docs for.
Use the sidebar, the search box, or the :ref:`index <genindex>` to find a symbol.

The import paths shown here are the supported ones. Version 1 public imports follow semantic versioning; see :doc:`../../API_REFERENCE` for the stability policy and
the semantics that span more than one class.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Page
     - Covers

   * - :doc:`core`
     - :class:`~medaugmentx.MedVolume`, the :class:`~medaugmentx.Transform` base
       class, the pipeline containers, and the coordinate maps used to write
       custom spatial transforms.

   * - :doc:`transforms`
     - Every shipped augmentation, grouped by modality. Start from the
       :doc:`catalog <../catalog>` if you are still choosing one.

   * - :doc:`pipelines`
     - Modality presets, JSON/YAML serialisation, the transform registry, and
       pipeline inspection.

   * - :doc:`validation`
     - Plausibility checks on augmented volumes, and the guard that enforces
       them inside a pipeline.

   * - :doc:`io`
     - DICOM and NIfTI readers, framework adapters for PyTorch, MONAI, and
       TorchIO, and the synthetic phantoms used throughout the docs.

.. toctree::
   :hidden:

   core
   transforms
   pipelines
   validation
   io
