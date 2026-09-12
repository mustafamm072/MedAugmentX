Transform catalog
=================

Every augmentation MedAugmentX ships, grouped by what it models. Use this page
to answer "which transform do I want, and will it move my annotations?"; follow
a name through to the :doc:`API reference <api/transforms>` for its full
docstring, parameter meanings, and defaults.

All transforms are importable from one flat namespace:

.. code-block:: python

   from medaugmentx.transforms import BiasField, MRIMotion, RandomAffine

.. rubric:: Reading the tables

**Parameters** lists the arguments that shape the effect. Two more are on every
transform and are left out of the tables:

``p``
   Probability of applying the transform at all, in ``[0, 1]``. Default ``1.0``.

``seed``
   Integer seed or :class:`numpy.random.Generator`. Inside a container, children
   are seeded from the parent instead, each with its own stream.

**Targets** says what happens to ``keypoints`` and ``bboxes`` on the volume:

``warped``
   Coordinates are mapped through the same geometry as the pixels. Boxes are
   mapped by their corners and re-bounded to an axis-aligned box, so they stay
   valid under rotation. Targets are never silently dropped — after a crop, call
   :meth:`~medaugmentx.MedVolume.remove_out_of_bounds_targets` to prune the ones
   that left the frame.

``unchanged``
   The pixel grid does not move, so coordinates are passed through untouched.

Masks follow the image for every spatial transform and are resampled with
nearest-neighbour interpolation, so label values are never blended into new
ones.

.. note::

   This page is generated from the installed package at build time — the
   transform list, summaries, parameters, and target behaviour are all read out
   of the code rather than transcribed, so they cannot fall out of date.

.. mx-transform-catalog::

Ready-made pipelines
--------------------

Each modality has a preset that composes a sensible starting point. They return
an ordinary :class:`~medaugmentx.Compose`, so you can inspect, edit, serialise,
or use one as a template for your own.

.. code-block:: python

   from medaugmentx.presets import ct_pipeline

   pipeline = ct_pipeline(seed=0)
   print(pipeline)

See :doc:`api/pipelines` for what each preset contains.
