Transforms
==========

.. code-block:: python

   from medaugmentx.transforms import BiasField, MRIMotion, RandomAffine

Full documentation for every shipped augmentation. For a side-by-side view of
what each one simulates and whether it moves your annotations, see the
:doc:`transform catalog <../catalog>`.

Two arguments are common to all of them and are not repeated below unless a
transform treats them specially:

``p``
   Probability of applying the transform, in ``[0, 1]``. When it does not fire,
   the input volume is returned unchanged.

``seed``
   Integer seed or :class:`numpy.random.Generator`. Containers override this for
   their children, giving each an independent stream derived from the parent.

Ranges follow one convention throughout: a scalar means a symmetric range around
zero or a fixed value, depending on the parameter, and a ``(low, high)`` tuple
is always explicit. Each docstring states which applies.

.. mx-transform-api::
