"""Sphinx configuration for the MedAugmentX documentation site.

Two source kinds feed the site:

* The Markdown guides in the repository, rendered by MyST so there is never a
  second copy to keep in sync.
* Generated reference pages built from the installed package — ``autodoc`` for
  docstrings and the ``mx_catalog`` extension for the transform catalog. Nothing
  in the reference is hand-transcribed, so it cannot drift from the code.

Sphinx is invoked with the repository root as the source directory
(``sphinx -c docs/sphinx .``) so the top-level Markdown files can be included
without copying them; ``include_patterns`` keeps everything else out.
"""
import os
import sys
from importlib.metadata import version as package_version

# Local extensions live beside this file, not on the default path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "_ext")))

project = "MedAugmentX"
author = "MedAugmentX Contributors"
release = package_version("medaugmentx")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "mx_catalog",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "docs/sphinx/index"
include_patterns = [
    "README.md", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md",
    "docs/*.md", "docs/sphinx/**.rst", "examples/README.md", "benchmarks/README.md",
    "notebooks/README.md",
]

# -- Markdown ---------------------------------------------------------------
myst_heading_anchors = 4
myst_enable_extensions = ["colon_fence", "deflist"]

# -- API reference ----------------------------------------------------------
# Signatures come from the code, prose from Google-style docstrings. Members are
# emitted in source order rather than alphabetically, because the transforms and
# containers are written with their most important methods first.
autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_class_signature = "mixed"
autodoc_inherit_docstrings = False
# Types belong with the parameter descriptions, not crammed into the signature.
# Signatures then read as the call you would actually write — name and default —
# which matters most for the transforms, where the defaults are the useful part.
# It also sidesteps Sphinx mangling the MedVolume dataclass signature, where an
# in-signature ``tuple[float, ...]`` gets split at its comma into two arguments.
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
# The transform modules share a handful of small annotation aliases. Rendering
# them by name keeps signatures readable instead of expanding every Union.
autodoc_type_aliases = {
    "Range": "Range",
    "IntRange": "IntRange",
    "SeedLike": "SeedLike",
    "ChannelDim": "ChannelDim",
    "PointMap": "PointMap",
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
# Render "Attributes:" sections as field-list entries rather than standalone
# attribute directives. Emitting directives would describe each attribute twice
# — once from the class docstring, once from autodoc's own member scan — which
# is a warning, and a build failure under CI's -W.
napoleon_use_ivar = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}
# Resolving inventories over the network would make the build depend on three
# external sites. CI builds with -W, so a slow or unreachable host must not be
# able to fail a release; a missing link degrades to plain text instead.
intersphinx_timeout = 10
intersphinx_disabled_reftypes = ["std:doc"]

# -- HTML output ------------------------------------------------------------
# Furo supplies the sidebar navigation, per-page contents, and light/dark
# switching that the stock theme has no equivalent for.
html_theme = "furo"
html_title = f"MedAugmentX {release}"
html_show_sourcelink = False
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# Copied verbatim to the output root: a redirect to the real root document, and
# .nojekyll so a Jekyll-processed host does not strip Sphinx's _static/ assets.
# Sphinx builds with the repository root as the source directory, so the root
# document lands at docs/sphinx/index.html rather than at the top of the output
# tree; without the redirect the site's front door 404s. Resolved relative to
# this config directory, not the source root.
html_extra_path = ["_extra"]
html_theme_options = {
    "source_repository": "https://github.com/mustafamm072/MedAugmentX/",
    "source_branch": "main",
    # Sphinx's source directory *is* the repository root here, so document names
    # already carry their full repo-relative path. Anything else would send the
    # "Edit this page" links to a doubled-up path such as docs/docs/sphinx/….
    "source_directory": "",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/mustafamm072/MedAugmentX",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0"'
                ' viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0'
                ' 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38'
                ' 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94'
                '-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21'
                ' 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95'
                ' 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21'
                ' 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82'
                ' 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87'
                ' 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01'
                ' 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z">'
                "</path></svg>"
            ),
            "class": "",
        },
    ],
}
