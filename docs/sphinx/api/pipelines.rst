Presets, serialisation & inspection
===================================

Everything for turning a pipeline into something you can record, reload, and
read back months later.

Modality presets
----------------

.. code-block:: python

   from medaugmentx.presets import ct_pipeline, dbt_pipeline, dxr_pipeline, mri_pipeline

Each factory returns a plain :class:`~medaugmentx.Compose`. They are starting
points chosen to be conservative, not tuned policies — inspect one, then adjust
it for your data.

.. automodule:: medaugmentx.presets
   :members:

Serialisation
-------------

.. code-block:: python

   from medaugmentx.serialization import from_json, from_yaml, to_json, to_yaml

A pipeline round-trips through JSON or YAML, which makes the exact augmentation
policy an artifact you can commit next to a set of results. Reconstruction only
ever instantiates registered ``Transform`` subclasses, and rejects oversized or
deeply nested documents.

.. warning::

   A saved integer seed restarts the *original* draw sequence — it does not
   capture the RNG state at the moment of saving. Reloading mid-run therefore
   replays from the beginning of that stream rather than continuing.

   Registering a custom transform makes its constructor reachable by name from
   any document you load, so treat pipeline files from outside your project the
   way you would treat any other executable input. See the
   :doc:`security policy </SECURITY>`.

.. automodule:: medaugmentx.serialization
   :members: to_json, from_json, to_yaml, from_yaml, from_dict, register_transform

.. py:data:: medaugmentx.serialization.REGISTRY

   Mapping of transform class name to class, consulted by
   :func:`~medaugmentx.serialization.from_dict`. Every shipped transform is
   registered at import time. Custom classes are normally added with the
   :func:`~medaugmentx.serialization.register_transform` decorator, which
   validates the class and refuses to silently replace an existing name;
   assigning into the mapping directly also works.

Inspection
----------

.. code-block:: python

   from medaugmentx import iter_pipeline, pipeline_summary

``pipeline_summary`` renders a nested pipeline as a stable, readable tree —
useful in experiment logs and review notes. ``iter_pipeline`` exposes the same
traversal programmatically, for instance to assert in a test that no
mask-destroying transform ended up in a segmentation pipeline.

.. automodule:: medaugmentx.inspection
   :members:
