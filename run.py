"""Single entrypoint for the pipeline phases."""

from __future__ import annotations

import argparse
import importlib
import json
import sys

from src.config import Config

PHASES = ("1", "2", "3", "4")


def main() -> int:
    parser = argparse.ArgumentParser(description="Healthcare cost prediction pipeline")
    parser.add_argument("phase", choices=list(PHASES) + ["all"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true", help="rebuild cached interim data")
    args = parser.parse_args()

    config = Config.load(args.config)
    selected = list(PHASES) if args.phase == "all" else [args.phase]

    for name in selected:
        module = importlib.import_module(f"src.phase{name}")
        print(f"running phase {name}")
        summary = module.run(config, force=args.force)
        print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
