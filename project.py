#!/usr/bin/env python3
"""Dispatch to isolated route tools, without importing a GPU framework."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"--help", "-h"}:
        print("Usage: python project.py {harp|larp|r0} [route arguments]")
        print("       python project.py reproduce {plan|run|evaluate} [arguments]")
        print("       python project.py verify")
        print("See docs/RUN_REPRODUCTION.md for independent baseline, main, ablation and first-reload execution.")
        return 0
    route = args.pop(0)
    if route == "verify":
        script = ROOT / "tools" / "verify_release.py"
    elif route in {"harp", "larp"}:
        script = ROOT / "routes" / route / "cli.py"
    elif route in {"reproduce", "r0"}:
        script = ROOT / "reproduce.py"
        if route == "r0" and args and args[0] not in ("--help", "-h"):
            if args[0] == "train":
                args[0] = "run"
            if args[0] in ("plan", "run"):
                args[1:1] = ["--route", "r0"]
    else:
        print(f"Unknown route: {route!r}. Expected harp, larp, r0, reproduce, or verify.", file=sys.stderr)
        return 2
    if not script.is_file():
        print(f"Incomplete release: missing {script.relative_to(ROOT)}", file=sys.stderr)
        return 2
    return subprocess.run([sys.executable, str(script), *args], cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
