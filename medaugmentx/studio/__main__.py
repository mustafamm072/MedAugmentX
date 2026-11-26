"""``python -m medaugmentx.studio`` — same options as ``medaugmentx studio``."""

import sys

from medaugmentx.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["studio", *sys.argv[1:]]))
