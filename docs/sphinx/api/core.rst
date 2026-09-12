Core
====

.. code-block:: python

   from medaugmentx import Compose, MedVolume, OneOf, SomeOf, Transform

The data container
------------------

``MedVolume`` is the single type that flows through every pipeline. It is
immutable in practice: transforms return a new volume via
:meth:`~medaugmentx.MedVolume.replace` or :meth:`~medaugmentx.MedVolume.warp`
rather than mutating the one they were given.

Coordinates in ``keypoints`` and ``bboxes`` use **array-index order** — the same
order you would use to index ``image``, so ``(z, y, x)`` for 3D and ``(y, x)``
for 2D — and boxes are laid out low corner first, ``[min…, max…]``.

.. autoclass:: medaugmentx.MedVolume
   :members:

The transform contract
----------------------

Every augmentation subclasses ``Transform``. Subclasses implement
:meth:`~medaugmentx.Transform.apply` and draw all randomness from ``self.rng``;
the base class handles probability gating, seeding, and serialisation, so a
custom transform gets those for free.

.. autoclass:: medaugmentx.Transform
   :members:
   :special-members: __call__

Pipeline containers
-------------------

Containers are themselves transforms, so they nest freely. Each one seeds its
children from its own generator rather than sharing a stream, which keeps two
identically-seeded transforms in the same pipeline from drawing the same
numbers.

.. autoclass:: medaugmentx.Compose

.. autoclass:: medaugmentx.OneOf

.. autoclass:: medaugmentx.SomeOf

Coordinate maps
---------------

These are the building blocks behind the *Targets* column of the
:doc:`catalog <../catalog>`. You need them only when writing a spatial transform
of your own: build a point map describing how coordinates move, then hand it to
:meth:`~medaugmentx.MedVolume.warp` along with the resampled image, and
keypoints and boxes are carried along for you.

.. automodule:: medaugmentx.core.geometry
   :members: map_keypoints, map_bboxes, flip_map, affine_map, translate_map, scale_map, displacement_map
