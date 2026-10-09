#!/usr/bin/env python3
"""Plan exact external assets; download only explicitly selected official CLIP files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
HF_REVISION = "ceb3112337fb6e1bf71534e355ca5569b63f19a6"
HF_SOURCE = "https://huggingface.co/zhengli97/prompt_learning_dataset"
TEACHER_SOURCES = [
    "https://pan.baidu.com/s/1KNJ1mhNKoxdSli4ZldeZUg?pwd=mjf4",
    "https://terabox.com/s/1X4mxJtSaR8W2lrK5bsrCkg",
    "https://drive.google.com/drive/folders/1OdQ9WauZmYAzVSUTTw7tIKKChyECIS5B?usp=sharing",
]
# IDs and archive names are transcribed from PromptKD's published README/DATASETS.md.
# The source links are not assertions that a new download has passed the manifest.
DATASET_SOURCES = {
    "caltech-101": ("caltech-101.zip", "1hyarUivQE36mY6jSomru6Fjd-JzwcCzN"),
    "dtd": ("dtd.zip", "1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x"),
    "eurosat": ("eurosat.zip", "1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o"),
    "food-101": ("food101.zip", "1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl"),
    "oxford_flowers": ("oxford_flowers.zip", "1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT"),
    "oxford_pets": ("oxford_pets.zip", "1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs"),
    "stanford_cars": ("stanford_cars.zip", "1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT"),
    "sun397": ("sun397.zip", "1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq"),
    "ucf101": ("ucf101.zip", "1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y"),
    "fgvc_aircraft": ("fgvc_aircraft.zip", None),
}
CLIP_URLS = {
    "clip/ViT-B-16.pt": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "clip/ViT-L-14.pt": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def classify(entry):
    if entry["kind"] == "data_metadata":
        return "splits"
    return "clip" if entry["relative_path"] in CLIP_URLS else "teachers"


def asset_plan(entry, data_root, pretrained_root):
    relative = PurePosixPath(entry["relative_path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Invalid relative path in external asset manifest")
    group = classify(entry)
    target_root = pretrained_root if entry["kind"] == "pretrained" else data_root
    row = {"group": group, "relative_path": relative.as_posix(),
           "target": str(target_root / relative), "bytes": entry["bytes"],
           "sha256": entry["sha256"], "download_url": CLIP_URLS.get(relative.as_posix())}
    if group == "clip":
        # Avoid silently using an edited hash with an unrelated hardcoded URL.
        if row["download_url"].split("/")[-2] != entry["sha256"]:
            raise ValueError("CLIP manifest hash differs from the official URL")
        row["source"] = "https://github.com/openai/CLIP/blob/main/clip/clip.py"
        row["availability"] = "official_direct_url; new download still requires size/SHA validation"
    elif group == "teachers":
        row["source"] = TEACHER_SOURCES
        row["availability"] = "manual; exact-byte public provenance not established by historical audit"
    else:
        archive, drive_id = DATASET_SOURCES[relative.parts[0]]
        row["source"] = f"{HF_SOURCE}/blob/{HF_REVISION}/{archive}"
        row["archive_url"] = f"{HF_SOURCE}/resolve/{HF_REVISION}/{archive}?download=true"
        row["split_source"] = (f"https://drive.google.com/file/d/{drive_id}/view?usp=sharing"
                               if drive_id else "Aircraft metadata is included in the dataset archive")
        row["availability"] = "manual; validate exact metadata after extraction; image bytes not checked here"
    return row


def existing_state(target, row):
    if target.is_symlink():
        raise ValueError(f"Refusing a symlink destination: {target}")
    if not target.exists():
        return None
    if target.is_file() and target.stat().st_size == row["bytes"] and digest(target) == row["sha256"]:
        return "already_verified"
    raise ValueError(f"Existing destination differs; no overwrite: {target}")


def download_verified(row, timeout=60):
    """Stream bytes, retain failed partials, and publish without replacing a destination."""
    if row["group"] != "clip" or row["download_url"] != CLIP_URLS.get(row["relative_path"]):
        raise ValueError("Only the two documented official CLIP URLs may be downloaded")
    target = Path(row["target"]).expanduser().absolute()
    state = existing_state(target, row)
    if state:
        return {"relative_path": row["relative_path"], "status": state}
    for parent in target.parents:
        if parent.is_symlink():
            raise ValueError(f"Refusing symlink destination parent: {parent}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise ValueError(f"Partial download retained from earlier work; inspect it first: {partial}")
    if shutil.disk_usage(target.parent).free < row["bytes"]:
        raise ValueError(f"Insufficient free bytes for {target}")
    observed = hashlib.sha256()
    count = 0
    request = Request(row["download_url"], headers={"User-Agent": "promptkd-harp-larp-asset-preparation/1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200 or not response.geturl().startswith("https://"):
                raise ValueError("Expected a successful HTTPS file response")
            with partial.open("xb") as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    count += len(block)
                    if count > row["bytes"]:
                        raise ValueError("Response exceeds the registered byte count")
                    stream.write(block)
                    observed.update(block)
                stream.flush()
                os.fsync(stream.fileno())
        if count != row["bytes"] or observed.hexdigest() != row["sha256"]:
            raise ValueError("Downloaded file size/SHA256 does not match the historical asset")
        # Same-directory hardlink creation is exclusive; unlike replace(), it never
        # overwrites a target created by another process while downloading.
        os.link(partial, target)
        partial.unlink()
    except Exception as exc:
        raise RuntimeError(f"Download not installed; existing data unchanged; inspect {partial}: {exc}") from exc
    return {"relative_path": row["relative_path"], "status": "downloaded_size_sha256_verified",
            "bytes": count, "sha256": observed.hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("all", "clip", "teachers", "splits"), default="all")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--pretrained-root", type=Path, default=Path("pretrained"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Default: no network, directories or file writes")
    mode.add_argument("--download", action="store_true", help="Only valid with --group clip; explicitly fetch both CLIP files")
    parser.add_argument("--timeout", type=int, default=60, help="Per-network-operation timeout, seconds")
    args = parser.parse_args(argv)
    if args.download and args.group != "clip":
        parser.error("Use --group clip --download; teachers and dataset metadata require the documented manual sources")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    manifest = json.loads((ROOT / "metadata/external_assets.json").read_text(encoding="utf-8"))
    rows = [asset_plan(e, args.data_root.expanduser(), args.pretrained_root.expanduser())
            for e in manifest["assets"] if args.group == "all" or classify(e) == args.group]
    result = {"status": "dry_run_no_network_no_writes", "assets": rows,
              "image_inventory_verified": False, "teacher_public_exact_bytes_verified": False}
    if args.download:
        result["downloads"] = []
        try:
            for row in rows:
                result["downloads"].append(download_verified(row, args.timeout))
        except (OSError, ValueError, RuntimeError) as exc:
            result.update(status="failed_no_overwrite", error=str(exc))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1
        result["status"] = "selected_clip_assets_verified"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
