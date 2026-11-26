"""MedAugmentX Policy Studio — design, preview and stress-test augmentation policies.

A local, dependency-free web application. Start it from a terminal::

    medaugmentx studio --phantom mri

or from Python / a notebook::

    from medaugmentx.studio import launch

    studio = launch(policy, volume, block=False)
    ...  # tune the policy in the browser
    policy = studio.current_policy()
    studio.stop()

The server binds to localhost, protects its API with a per-session token and
loads no remote assets, so images stay on your machine.
"""

from medaugmentx.studio.server import StudioServer, launch
from medaugmentx.studio.session import PolicyError, StudioSession, python_snippet

__all__ = ["PolicyError", "StudioServer", "StudioSession", "launch", "python_snippet"]
