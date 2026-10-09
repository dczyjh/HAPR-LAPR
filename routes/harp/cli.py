#!/usr/bin/env python3
"""HARP configuration inspection and isolated reproduction entry points."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
BACKEND = dict(benchmark=True, deterministic=False, deterministic_algorithms=False,
               flash=True, efficient=False, math=True)
VARIANTS = {"r1": ([8, 9, 10, 11], 32, 0), "harp": ([8, 9, 10, 11], 32, 199808),
            "mid4_d32": ([4, 5, 6, 7], 32, 199808),
            "last4_d16": ([8, 9, 10, 11], 16, 101440),
            "last4_d64": ([8, 9, 10, 11], 64, 396544)}
DATASETS = ("caltech101", "oxford_pets", "stanford_cars", "oxford_flowers", "food101",
            "fgvc_aircraft", "sun397", "dtd", "eurosat", "ucf101")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_science(cfg, dataset, seed, variant):
    """Check preserved protocol fields; do not apply runtime defaults."""
    kd = cfg["TRAINER"]["PROMPTKD"]
    adaptation = kd["ADAPTATION"]
    layers, width, _ = VARIANTS[variant]
    assert cfg["SEED"] == seed and seed in (1, 2, 3)
    assert cfg["TRAINER"]["NAME"] == "PromptKD" and cfg["TRAINER"]["MODAL"] == "base2novel"
    assert cfg["OPTIM"]["MAX_EPOCH"] == 20 and cfg["OPTIM"]["NAME"] == "sgd"
    assert cfg["OPTIM"]["LR"] == .005 and cfg["OPTIM"]["MOMENTUM"] == .9
    assert cfg["OPTIM"]["WEIGHT_DECAY"] == .0005
    assert cfg["OPTIM"]["LR_SCHEDULER"] == "cosine" and cfg["OPTIM"]["WARMUP_RECOUNT"]
    assert cfg["OPTIM"]["WARMUP_CONS_LR"] == 1e-5 and cfg["OPTIM"]["WARMUP_EPOCH"] == 1
    assert not cfg["OPTIM"]["SGD_NESTEROV"] and cfg["OPTIM"]["SGD_DAMPNING"] == 0
    assert kd["PREC"] == "fp16" and kd["TEMPERATURE"] == 1 and kd["CE_WEIGHT"] == 0
    assert kd["KD_WEIGHT"] == (200 if dataset in ("fgvc_aircraft", "oxford_flowers", "dtd") else 1000)
    assert not kd["REPRODUCIBLE"] and not kd["DEVELOPMENT"]
    assert adaptation["TYPE"] == ("none" if variant == "r1" else "harp")
    assert adaptation["LAYERS"] == layers and adaptation["HARP_DIM"] == width
    assert adaptation["HARP_SCALE"] == .001 and adaptation["REP_WEIGHT"] == 0
    assert adaptation["LR_MULT"] == (1 if variant == "r1" else .1)
    assert cfg["DATALOADER"]["TRAIN_X"]["BATCH_SIZE"] == 8
    assert cfg["DATALOADER"]["TEST"]["BATCH_SIZE"] == 100 and cfg["DATALOADER"]["NUM_WORKERS"] == 8
    assert cfg["TEST"]["FINAL_MODEL"] == "best_val" and cfg["DATASET"]["NUM_SHOTS"] == 0
    assert cfg["RESUME"] == "" and cfg["USE_CUDA"]


def check():
    manifest = read(ROOT / "provenance/generated_files.json")
    for relative, record in manifest.items():
        path = ROOT / relative
        assert path.is_file() and not path.is_symlink(), ("missing or symlink", relative)
        assert sha_file(path) == record["sha256"], ("hash mismatch", relative)
    syntax = 0
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        syntax += 1
    rows = read(ROOT / "configs/index.json")
    expected = {(d, s, v) for d in DATASETS for s in (1, 2, 3) for v in VARIANTS}
    assert {(r["dataset"], r["seed"], r["variant"]) for r in rows} == expected
    assert len(rows) == len(expected) == 150
    for row in rows:
        validate_science(read(ROOT / row["portable"]), row["dataset"], row["seed"], row["variant"])
        assert sha_file(ROOT / row["recorded"]) == row["recorded_sha256"]
    return dict(status="passed_static_offline", source_files=len(manifest), python_files=syntax,
                recorded_configs=len(rows), portable_training_enabled=True, cuda_run_validated=False,
                note="No dataset reads, model evaluation, GPU gates, training or score reproduction.")


def plan(args):
    row = next(r for r in read(ROOT / "configs/index.json")
               if (r["dataset"], r["seed"], r["variant"]) == (args.dataset, args.seed, args.variant))
    cfg = copy.deepcopy(read(ROOT / row["portable"]))
    validate_science(cfg, args.dataset, args.seed, args.variant)
    replacements = {"DATA_ROOT": args.data_root, "CLIP_ROOT": args.clip_root,
                    "TEACHER_ROOT": args.teacher_root, "OUTPUT_DIR": args.output}
    replacements = {key: str(Path(value).expanduser().resolve()) if value else "${" + key + "}"
                    for key, value in replacements.items()}
    cfg["DATASET"]["ROOT"] = replacements["DATA_ROOT"]
    cfg["TRAINER"]["PROMPTKD"]["WEIGHTS_ROOT"] = replacements["CLIP_ROOT"]
    cfg["TRAINER"]["PROMPTKD"]["TEACHER_ROOT"] = replacements["TEACHER_ROOT"]
    cfg["OUTPUT_DIR"] = replacements["OUTPUT_DIR"]
    return dict(status="configuration_plan_only", dataset=args.dataset, seed=args.seed, variant=args.variant,
                recorded_source=row["recorded"], recorded_sha256=row["recorded_sha256"],
                added_parameters=VARIANTS[args.variant][2], backend=BACKEND,
                common_schedule="projector epoch2 linear 1e-5 -> .005; momentum retained",
                added_schedule=None if args.variant == "r1" else "hold epoch1 with grad=None; epoch2 1e-6 -> .0005; epoch3+ common LR * .1",
                initialization="from original CLIP/teacher, not trained HARP checkpoint",
                evaluation="first strict highest Base-test checkpoint; first independent reload; original row-scale hard gate",
                portable_training_enabled=True, cuda_run_validated=False, resolved_config=cfg)


def preflight(args):
    result = plan(args)
    if not all((args.data_root, args.clip_root, args.teacher_root, args.output)):
        raise ValueError("preflight requires --data-root --clip-root --teacher-root --output")
    cfg = result["resolved_config"]
    assert Path(cfg["DATASET"]["ROOT"]).is_dir(), "Dataset root missing"
    assert not Path(cfg["OUTPUT_DIR"]).exists(), "Output must be a new nonexisting path"
    assets = read(ROOT / "provenance/external_assets.json")
    checked = []
    for asset in assets["clip"]:
        path = Path(args.clip_root).expanduser() / asset["relative_path"]
        assert sha_file(path) == asset["sha256"], ("CLIP hash mismatch", asset["relative_path"])
        checked.append(asset["relative_path"])
    teacher = next(r for r in assets["teachers"] if r["dataset_name"] == cfg["DATASET"]["NAME"])
    assert sha_file(Path(args.teacher_root).expanduser() / teacher["relative_path"]) == teacher["sha256"]
    checked.append(teacher["relative_path"])
    result.update(status="passed_external_weight_hashes_only", checked_assets=checked,
                  dataset_identity_verified=False, gpu_gate_verified=False)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("train", "run", "evaluate"):
        command = "run" if argv[0] in ("train", "run") else "evaluate"
        arguments = ["--output-root" if x == "--output" else x for x in argv[1:]]
        route = [] if command == "evaluate" else ["--route", "harp"]
        return subprocess.run([sys.executable, "-B", str(ROOT.parents[1] / "reproduce.py"), command, *route, *arguments], check=False).returncode
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check", "preflight", "train", "evaluate"))
    parser.add_argument("--dataset", choices=DATASETS, default="caltech101")
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--variant", choices=VARIANTS, default="harp")
    parser.add_argument("--data-root")
    parser.add_argument("--clip-root")
    parser.add_argument("--teacher-root")
    parser.add_argument("--output")
    parser.add_argument("--cpu-tests", action="store_true", help="Run tensor/scheduler unit tests, not GPU training")
    args = parser.parse_args(argv)
    result = check() if args.command == "check" else preflight(args) if args.command == "preflight" else plan(args)
    if args.cpu_tests:
        if args.command != "check":
            parser.error("--cpu-tests is supported only with check")
        subprocess.run([sys.executable, "-B", str(ROOT / "tests/test_offline.py")], check=True)
        subprocess.run([sys.executable, "-B", str(ROOT / "tests/test_portable.py")], check=True)
        result["cpu_unit_tests"] = "passed; not FP16 CUDA or end-to-end validation"
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
