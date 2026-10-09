#!/usr/bin/env python3
"""LARP archive/configuration inspection and isolated reproduction entry points."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve_config(dataset, variant, seed, data_root, clip_root, teacher_root, output_root):
    protocol = read_json("protocol/protocol.json")
    variants = read_json("protocol/variants.json")
    if dataset not in protocol["datasets"] or variant not in variants or seed not in protocol["seeds"]:
        raise ValueError("Dataset, variant and seed must be members of the frozen protocol")
    cfg = copy.deepcopy(read_json("protocol/resolved_templates.json")[dataset]["r1" if variant == "r1" else "o4"])
    cfg["DATASET"]["ROOT"] = str(data_root)
    cfg["TRAINER"]["PROMPTKD"]["WEIGHTS_ROOT"] = str(clip_root)
    cfg["TRAINER"]["PROMPTKD"]["TEACHER_ROOT"] = str(teacher_root)
    cfg["OUTPUT_DIR"] = str(Path(output_root) / dataset / f"seed_{seed}" / variant)
    cfg["SEED"] = seed
    if variant != "r1":
        spec = variants[variant]
        cfg["TRAINER"]["PROMPTKD"]["ADAPTATION"].update(LORA_R=spec["rank"], LAYERS=spec["layers"])
    return cfg


def plan(args):
    protocol = read_json("protocol/protocol.json")
    variants = read_json("protocol/variants.json")
    specs = read_json("protocol/datasets.json")
    datasets = args.dataset or protocol["datasets"]
    seeds = args.seed or protocol["seeds"]
    requested_variants = list(variants) if args.all_variants else (args.variant or ["r1", "o4-r2"])
    jobs = []
    for dataset in datasets:
        for seed in seeds:
            for variant in requested_variants:
                cfg = resolve_config(dataset, variant, seed, args.data_root, args.clip_root,
                                     args.teacher_root, args.output_root)
                row = {"dataset": dataset, "seed": seed, "variant": variant, "epochs": 20,
                       "kd_weight": specs[dataset]["kd_weight"],
                       "added_parameters": variants[variant]["added_parameters"],
                       "trainable_parameters": 1013760 + variants[variant]["added_parameters"],
                       "resolved_config_sha256": canonical_sha256(cfg), "output_dir": cfg["OUTPUT_DIR"]}
                if args.resolved:
                    row["resolved_config"] = cfg
                jobs.append(row)
    return {"status": "plan_only_no_execution", "route": "larp", "portable_training_enabled": True,
            "job_count": len(jobs), "epoch_count": 20 * len(jobs), "jobs": jobs,
            "backend": protocol["backend"], "cuda_run_validated": False}


def check_sources():
    inventory = read_json("provenance/source_inventory.json")
    errors = []
    rows = inventory["files"] + inventory["workers"] + inventory["supplementary_assets"]
    for row in rows:
        path = ROOT / row["release_path"]
        if path.is_symlink() or not path.is_file() or file_sha256(path) != row["sha256"]:
            errors.append({"path": row["release_path"], "reason": "missing_or_source_hash_mismatch"})
    manifest = ROOT / "provenance/final_scientific_manifest.json"
    if file_sha256(manifest) != inventory["scientific_manifest_sha256"]:
        errors.append({"path": "provenance/final_scientific_manifest.json", "reason": "manifest_hash_mismatch"})
    return {"status": "passed" if not errors else "failed", "checked_files": len(rows),
            "frozen_scientific_sources": len(inventory["files"]), "errors": errors}


def check_assets(args):
    specs = read_json("protocol/datasets.json")
    checked, errors, cache = [], [], {}
    if not Path(args.data_root).is_dir():
        errors.append({"path": args.data_root, "reason": "data_root_missing"})
    for dataset, spec in specs.items():
        for key, expected in spec["assets_sha256"].items():
            path = (Path(args.teacher_root) / spec["dataset_class"] / "VLPromptLearner/model-best.pth.tar"
                    if key == "teacher" else Path(args.clip_root) / key)
            if path not in cache:
                cache[path] = file_sha256(path) if path.is_file() else None
            passed = cache[path] == expected
            checked.append({"dataset": dataset, "asset": key, "path": str(path), "passed": passed})
            if not passed:
                errors.append({"path": str(path), "reason": "missing_or_asset_hash_mismatch"})
    return {"status": "passed" if not errors else "failed", "assets": checked, "errors": errors,
            "dataset_reader_and_split_verified": False,
            "note": "Only the data root exists check and teacher/CLIP file hashes are performed; no dataset or checkpoint is loaded."}


def check_runtime():
    expected = read_json("environment/runtime.json")
    packages = {}
    for name in ("torch", "torchvision", "numpy", "Pillow", "yacs"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    errors = []
    for condition, description in (
        (platform.system() == "Linux", "Linux required by the historical execution environment"),
        (platform.python_version() == expected["python"], "Python version differs from the frozen environment"),
        (packages["torch"] == expected["torch"], "PyTorch version differs from the frozen environment"),
        (packages["torchvision"] == expected["torchvision"], "torchvision version differs from the frozen environment"),
    ):
        if not condition:
            errors.append(description)
    return {"status": "passed" if not errors else "failed", "python": platform.python_version(),
            "system": platform.system(), "packages": packages, "errors": errors,
            "gpu_backend_checked": False, "note": "Reads installed metadata only; never imports torch or initializes CUDA."}


def make_parser():
    protocol = read_json("protocol/protocol.json")
    variants = read_json("protocol/variants.json")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="Print a fixed protocol plan as JSON without creating files or models")
    p.add_argument("--dataset", action="append", choices=protocol["datasets"], help="Repeat to select datasets; default all ten")
    p.add_argument("--seed", action="append", type=int, choices=[1, 2, 3], help="Repeat to select seeds; default all three")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--variant", action="append", choices=list(variants), help="Repeat to select variants; default R1 plus O4/rank2")
    group.add_argument("--all-variants", action="store_true", help="Include all five variants, 150 dataset/seed/variant roles")
    for name in ("data-root", "clip-root", "teacher-root", "output-root"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--resolved", action="store_true", help="Include complete resolved configuration dictionaries")
    c = sub.add_parser("check", help="Check archived source hashes; optional read-only asset and installed-version checks")
    c.add_argument("--runtime", action="store_true", help="Compare installed package metadata; no CUDA or model import")
    for name in ("data-root", "clip-root", "teacher-root"):
        c.add_argument("--" + name, help="Supply all three roots together to check teacher/CLIP hashes")
    sub.add_parser("train", help="Forward to reproduce.py run --route larp (use train --help for arguments)")
    sub.add_parser("evaluate", help="Forward to reproduce.py evaluate")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("train", "run", "evaluate"):
        command = "run" if argv[0] in ("train", "run") else "evaluate"
        route = [] if command == "evaluate" else ["--route", "larp"]
        return subprocess.run([sys.executable, "-B", str(ROOT.parents[1] / "reproduce.py"), command, *route, *argv[1:]], check=False).returncode
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        result, code = plan(args), 0
    else:
        roots = (args.data_root, args.clip_root, args.teacher_root)
        if any(roots) and not all(roots):
            parser.error("Asset checking requires --data-root, --clip-root and --teacher-root together")
        result = {"route": "larp", "sources": check_sources(), "portable_training_enabled": True, "cuda_run_validated": False}
        if all(roots):
            result["assets"] = check_assets(args)
        if args.runtime:
            result["runtime"] = check_runtime()
        failed = any(value.get("status") == "failed" for value in result.values() if isinstance(value, dict))
        result["status"] = "failed" if failed else "archive_checks_passed"
        code = 1 if failed else 0
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
