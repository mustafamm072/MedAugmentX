I/O, interop & phantoms
=======================

Getting data in, plugging into a training loop, and generating something to try
it all on.

Readers and writers
-------------------

.. code-block:: python

   from medaugmentx.io import load_dicom_series, load_nifti, save_nifti

These need optional dependencies — ``pip install "medaugmentx[io]"`` for both,
or ``[dicom]`` / ``[nifti]`` individually. The modules import their backend
lazily, so a core-only install can still import ``medaugmentx.io`` without
failing.

.. automodule:: medaugmentx.io
   :members:

Framework adapters
------------------

.. code-block:: python

   from medaugmentx.interop import MonaiMapTransform, TorchTransform, TorchIOTransform

The adapters wrap any MedAugmentX transform or pipeline so it can sit in a
dataset or transform chain from another framework. They accept the sample shapes
those frameworks use — arrays, tensors, ``(image, mask)`` pairs, dictionaries,
or a TorchIO ``Subject`` — and return the same kind they were given.

MedAugmentX works on single-channel 2D and 3D data. A singleton channel
dimension is stripped before augmentation and restored afterwards; ``channel_dim``
controls where the adapter expects to find it.

Importing an adapter never imports its framework. Torch, MONAI, and TorchIO are
touched only when a sample of that kind is actually passed through, so a single
``interop`` import costs nothing in a core-only environment.

.. automodule:: medaugmentx.interop
   :members:
   :member-order: bysource

Synthetic phantoms
------------------

.. code-block:: python

   from medaugmentx.phantoms import ct_phantom, dbt_phantom, mri_phantom, xray_phantom

Analytic phantoms used by the tutorials, examples, and tests. They are built
from geometric shapes and a fixed seed: no patient data, no downloads, no
network access, and byte-identical across machines — which makes them a
reasonable base for a regression test.

.. warning::

   Phantoms are illustrations, not validated models of anatomy or scanner
   physics. Do not draw clinical conclusions from how a transform looks on one.

.. automodule:: medaugmentx.phantoms
   :members: mri_phantom, ct_phantom, xray_phantom, dbt_phantom, DEFAULT_SIZE, DEFAULT_SEED
