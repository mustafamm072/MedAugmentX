"""Execute every tutorial notebook and fail if any cell raises.

The notebooks are committed without outputs, so this is what keeps them from
rotting: CI runs it on every push, and you can run it locally before a PR.

    python notebooks/run_notebooks.py            # all notebooks
    python notebooks/run_notebooks.py 01_mri_augmentation.ipynb

Requires the notebook extra:

    pip install -e ".[notebooks]"
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="notebook filenames (default: all)")
    parser.add_argument("--timeout", type=int, default=600, help="per-cell timeout in seconds")
    args = parser.parse_args()

    # Keep the matplotlib font cache out of the checkout so a CI runner with a
    # read-only or empty HOME still works.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "medaugmentx-matplotlib")
    )

    try:
        import nbformat
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError
    except ImportError as exc:  # pragma: no cover - depends on the environment
        print(
            f"Missing notebook dependencies ({exc}).\n"
            'Install them with: pip install -e ".[notebooks]"',
            file=sys.stderr,
        )
        return 2

    paths = [HERE / name for name in args.names] if args.names else sorted(HERE.glob("*.ipynb"))
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("No such notebook: " + ", ".join(p.name for p in missing), file=sys.stderr)
        return 2

    failures = 0
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        started = time.perf_counter()
        try:
            NotebookClient(
                notebook,
                timeout=args.timeout,
                kernel_name="python3",
                resources={"metadata": {"path": str(HERE)}},
            ).execute()
        except CellExecutionError as exc:
            failures += 1
            print(f"FAIL  {path.name}\n{exc}", file=sys.stderr)
        else:
            print(f"ok    {path.name}  ({time.perf_counter() - started:.1f}s)")

    if failures:
        print(f"\n{failures} notebook(s) failed.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
