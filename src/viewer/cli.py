"""
Command-line interface for the WSI viewer server.

Usage::

    wsi-view                              # default: 127.0.0.1:8000, browse from cwd
    wsi-view --host 0.0.0.0 --port 9000
    wsi-view --browse-dir /data/slides    # set the default browse directory
    wsi-view --reload                     # hot-reload on file changes (dev)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wsi-view",
        description="Start the WSI Viewer web server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Network interface to bind to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port to listen on.",
    )
    parser.add_argument(
        "--browse-dir",
        metavar="PATH",
        default=None,
        help="Default directory shown in the file browser on startup. "
             "Defaults to the current working directory.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload on source changes (development mode).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes (>1 disables --reload).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    return parser


def _get_version() -> str:
    try:
        from viewer import __version__
        return __version__
    except ImportError:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    browse_dir: Path | None = None
    if args.browse_dir is not None:
        browse_dir = Path(args.browse_dir).resolve()
        if not browse_dir.is_dir():
            parser.error(f"--browse-dir does not exist or is not a directory: {browse_dir}")

    # Build the app with the configured browse root, then hand off to uvicorn.
    # We import here (not at module level) so `wsi-view --help` is instant.
    import uvicorn
    from viewer.server import create_app

    application = create_app(default_browse_path=browse_dir)

    reload = args.reload and args.workers == 1  # uvicorn disallows reload + multiple workers

    print(f"  WSI Viewer  →  http://{args.host}:{args.port}")
    if browse_dir:
        print(f"  Browse root →  {browse_dir}")
    print()

    uvicorn.run(
        application,
        host=args.host,
        port=args.port,
        reload=reload,
        workers=args.workers if args.workers > 1 else None,
        log_level="info",
    )


if __name__ == "__main__":
    main()
