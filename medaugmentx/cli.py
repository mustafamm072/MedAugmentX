"""Command-line interface: ``medaugmentx <command>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from medaugmentx._version import __version__


def _studio(args: argparse.Namespace) -> int:
    from medaugmentx.studio.server import launch
    from medaugmentx.studio.session import PHANTOMS, StudioSession

    probe = StudioSession()
    try:
        if args.policy:
            with open(args.policy, encoding="utf-8") as handle:
                policy = probe.import_text(handle.read())["policy"]
        else:
            policy = None
        volume = None
        source = None
        if args.image:
            probe.load_path(args.image, args.mask, resample_mask=args.resample_mask)
            volume, source = probe.volume, probe.source
        elif args.phantom:
            probe.load_phantom(args.phantom)
            volume, source = probe.volume, probe.source
    finally:
        probe.close()
    if args.phantom and args.phantom not in PHANTOMS:  # pragma: no cover - argparse guards
        return 2
    launch(
        policy,
        volume,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        block=True,
        token=args.token,
        allow_remote=args.allow_remote,
        verbose=args.verbose,
        source=source,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medaugmentx", description="MedAugmentX command-line tools."
    )
    parser.add_argument("--version", action="version", version=f"medaugmentx {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    studio = commands.add_parser(
        "studio",
        help="Open the local policy studio in a web browser.",
        description="Design, preview and stress-test augmentation policies locally.",
    )
    source = studio.add_mutually_exclusive_group()
    source.add_argument("--image", help="NIfTI (.nii/.nii.gz), NumPy (.npy) or DICOM path.")
    source.add_argument(
        "--phantom", choices=["mri", "ct", "xray", "dbt"], help="Start from a synthetic phantom."
    )
    studio.add_argument("--mask", help="Registered label map for --image (NIfTI or .npy).")
    studio.add_argument(
        "--resample-mask",
        action="store_true",
        help="Resample a NIfTI mask on a different registered grid onto the image grid.",
    )
    studio.add_argument("--policy", help="JSON or YAML policy file to start editing.")
    studio.add_argument(
        "--host", default="127.0.0.1", help="Interface to bind (default: loopback)."
    )
    studio.add_argument("--port", type=int, default=0, help="Port (default: any free port).")
    studio.add_argument("--token", help="Fixed API token (default: random per session).")
    studio.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow binding a non-loopback --host. This exposes your images to the network.",
    )
    studio.add_argument("--no-browser", action="store_true", help="Do not open a browser.")
    studio.add_argument("--verbose", action="store_true", help="Log every request.")
    studio.set_defaults(handler=_studio)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mask and not args.image:
        parser.error("--mask requires --image")
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as exc:
        print(f"medaugmentx: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
