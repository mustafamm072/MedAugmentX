Validation & guards
===================

.. code-block:: python

   from medaugmentx import Guard, ValidationError, VolumeValidator

Aggressive augmentation can produce a volume that is no longer a plausible
image: intensities pushed to NaN, contrast collapsed to a constant, or a crop
that removed a labelled class entirely. ``VolumeValidator`` checks for those
conditions and ``Guard`` enforces the check inside a pipeline.

.. code-block:: python

   from medaugmentx import Compose, Guard
   from medaugmentx.transforms import AnatomicCrop, RandomAffine

   pipeline = Compose([
       Guard(RandomAffine(rotation=30), on_fail="retry", retries=3),
       Guard(AnatomicCrop(size=(128, 128)), on_fail="revert"),
   ], seed=0)

Some checks are *comparative*: they compare the augmented volume against the
volume as it was before the transform ran, and are skipped when no reference is
available. ``Guard`` supplies that reference automatically. Calling
:meth:`~medaugmentx.VolumeValidator.validate` yourself only runs them if you
pass ``reference=``.

``Guard`` is itself a :class:`~medaugmentx.Transform`, so it nests inside
containers and round-trips through serialisation along with the transform and
validator it wraps.

The validator
-------------

.. autoclass:: medaugmentx.VolumeValidator
   :members:

The guard
---------

.. autoclass:: medaugmentx.Guard

Results
-------

A report is truthy when the volume passed — that is, when it carries no
error-severity issues — so ``if not report:`` is the idiomatic check.

.. autoclass:: medaugmentx.ValidationReport
   :members:

.. autoclass:: medaugmentx.ValidationIssue
   :members:

.. autoclass:: medaugmentx.ValidationError
   :members:
