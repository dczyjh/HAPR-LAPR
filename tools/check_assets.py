#!/usr/bin/env python3
"""Verify registered external file bytes, without loading tensors or changing data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def check_entry(root: Path, entry: dict) -> dict:
    relative = PurePosixPath(entry["relative_path"])
    result = {"kind": entry["kind"], "relative_path": relative.as_posix()}
    if relative.is_absolute() or ".." in relative.parts:
        return dict(result, status="invalid_manifest_path")
    path = root / relative
    if not path.is_file():
        return dict(result, status="missing")
    try:
        if path.stat().st_size != entry["bytes"]:
            return dict(result, status="size_mismatch")
        if sha256(path) != entry["sha256"]:
            return dict(result, status="sha256_mismatch")
    except OSError as exc:
        return dict(result, status="read_error", error=str(exc))
    return dict(result, status="passed")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--pretrained-root", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT / "metadata/external_assets.json").read_text(encoding="utf-8"))
    results = [check_entry(args.pretrained_root if x["kind"] == "pretrained" else args.data_root, x)
               for x in manifest["assets"]]
    passed = all(x["status"] == "passed" for x in results)
    print(json.dumps({"status": "passed_registered_files_only" if passed else "failed",
                      "scope": manifest["scope"], "image_completeness_verified": False,
                      "gpu_verified": False, "files": results}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
