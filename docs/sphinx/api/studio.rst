Policy Studio
=============

.. code-block:: python

   from medaugmentx.studio import StudioServer, StudioSession, launch

The studio is a local web application for designing, previewing, and
stress-testing policies. See the :doc:`user guide <../../STUDIO>` for the
workflow; this page documents the Python entry points for notebooks, scripts,
and embedding.

Launching
---------

.. autofunction:: medaugmentx.studio.launch

.. autoclass:: medaugmentx.studio.StudioServer
   :members: url, start, serve_forever, stop, current_policy

Session state
-------------

``StudioSession`` holds the loaded volume and the edited policy, independently
of the HTTP layer. Policies travel in the canonical ``to_dict()`` form.

.. autoclass:: medaugmentx.studio.StudioSession
   :members: load_phantom, load_path, add_channel, remove_channel, state, compile,
      set_policy, current_policy, new_step, preset, preview, gallery, stress, record,
      export, import_text, close

.. autoexception:: medaugmentx.studio.PolicyError

.. autofunction:: medaugmentx.studio.python_snippet
