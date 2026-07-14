"""Command-line entry point for the initial ClawRL framework."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from clawrl.system import default_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clawrl",
        description="Autonomous RL training powered by coding agents.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the framework manifest as JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the initial framework manifest."""
    args = _parser().parse_args(argv)
    manifest = default_manifest()

    if args.json:
        print(json.dumps(manifest.as_dict(), indent=2, sort_keys=True))
        return 0

    print(f"{manifest.name} {manifest.version}")
    print(manifest.description)
    print("Automation domains:")
    for domain in manifest.domains:
        print(f"  - {domain.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
