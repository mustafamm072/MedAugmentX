# Working on the documentation

The published site is at **<https://mustafamm072.github.io/MedAugmentX/>**. This
page is for people changing it.

For a deterministic offline build, set `MEDAUGMENTX_DOCS_OFFLINE=1` before
running Sphinx. CI uses this mode; it disables external intersphinx inventory
fetches while retaining local API cross-references.

## How the site is put together

Sphinx is invoked with the **repository root** as its source directory:

```bash
python -m sphinx -W --keep-going -b html -c docs/sphinx . /tmp/medaugmentx-docs
```

That is deliberate. `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, and
`SECURITY.md` are rendered where they live instead of being copied into a docs
tree, so there is never a second version of them to keep in sync.
`include_patterns` in [`docs/sphinx/conf.py`](sphinx/conf.py) lists exactly
which files are picked up; a Markdown file not matched there is invisible to the
build.

Because the root document therefore lands at `docs/sphinx/index.html` rather
than at the top of the output, `docs/sphinx/_extra/index.html` is copied to the
output root as a redirect, and CI asserts that the redirect exists rather than
publishing a site whose front door 404s.

## Generated vs. hand-written

Anything describing the *code* is generated, so it cannot drift:

| Page | Source |
| --- | --- |
| [Transform catalog](sphinx/catalog.rst) | `.. mx-transform-catalog::` — reads `medaugmentx.transforms.__all__` |
| [API reference](sphinx/api/index.rst) | `autodoc` + `napoleon`, from the docstrings |
| Per-transform entries | `.. mx-transform-api::` — one `autoclass` per exported transform |

Both directives live in [`docs/sphinx/_ext/mx_catalog.py`](sphinx/_ext/mx_catalog.py).
They derive the summary from the first paragraph of each class docstring, the
parameter list from the constructor signature, and the "Targets" column from
whether the defining module uses `medaugmentx.core.geometry` — which is the only
way a transform can move keypoints and boxes. **Adding a transform to
`medaugmentx/transforms/__init__.py` is all it takes to appear in the docs.**

Everything else is prose written by hand: the guides in this directory, and the
landing page at [`docs/sphinx/index.rst`](sphinx/index.rst).

## The guides

| Document | Covers |
| --- | --- |
| [API examples](API_EXAMPLES.md) | Task-shaped snippets for common workflows |
| [API reference](API_REFERENCE.md) | Stability policy and the semantics that span several classes — not signatures, which are generated |
| [Architecture](ARCHITECTURE.md) | Layering, dependency boundaries, transform contracts, testing strategy |
| [Research guide](RESEARCH_GUIDE.md) | Replay, worker seeds, evaluation reporting, citation |
| [Commercial adoption](COMMERCIAL_ADOPTION.md) | Production readiness, intended use, validation, audit trail, privacy |
| [Milestones](MILESTONES.md) | Roadmap and v1.0 acceptance criteria |

## Making a change

```bash
pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html -c docs/sphinx . /tmp/medaugmentx-docs
open /tmp/medaugmentx-docs/index.html
```

`-W` turns warnings into errors, matching CI — a broken cross-reference or a
page missing from a toctree fails the build rather than shipping quietly. The
docs extras must be installed in the same environment as the package, since
`autodoc` imports `medaugmentx` to read its docstrings.

When adding a page:

1. Put `.rst` under `docs/sphinx/` (already covered by `include_patterns`), or
   Markdown in `docs/` .
2. Add it to one of the four `toctree` blocks in `docs/sphinx/index.rst` —
   Getting started, Reference, Guides, or Project. Every page must be in one, or
   `-W` will fail the build.
3. Rebuild and check the sidebar.

## Publishing

`.github/workflows/docs.yml` rebuilds and redeploys on every push to `main`, and
can be run by hand from the Actions tab (*Docs → Run workflow*). The repository
must keep **Settings → Pages → Source: GitHub Actions**; if that is switched
off, the build step still passes and only the deploy step fails.

`ci.yml` builds the same site on every push and pull request with `-W` and
uploads it as an artifact, so documentation warnings are caught on branches that
never deploy.
