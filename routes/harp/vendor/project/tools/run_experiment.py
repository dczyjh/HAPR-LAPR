"""Single source of truth for independent baseline/HARP/LARP runs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("caltech101", "oxford_pets", "stanford_cars", "oxford_flowers",
            "food101", "fgvc_aircraft", "sun397", "dtd", "eurosat", "ucf101")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("baseline", "harp", "larp"), required=True)
    p.add_argument("--dataset", choices=DATASETS + ("all",), default="caltech101")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    p.add_argument("--weights-root", default=str(ROOT / "clip"))
    p.add_argument("--teacher-root", default=str(ROOT / "teacher_model"))
    p.add_argument("--output-root", default=str(ROOT / "output" / "v3"))
    p.add_argument("--stage", choices=("benchmark", "development"), default="benchmark")
    p.add_argument("--protocol", choices=("fixed_last", "official_best_base"), default="fixed_last")
    p.add_argument("--kd-policy", choices=("command", "comment"), default="command",
                   help="command=1000 for all (released command); comment=200 for aircraft/flowers/dtd")
    p.add_argument("--kd-weight", type=float)
    p.add_argument("--rep-weight", type=float, default=0.0)
    p.add_argument("--adapt-lr-mult", type=float, default=1.0)
    p.add_argument("--lora-targets", nargs="+", choices=("q", "v"), default=["v"])
    p.add_argument("--harp-scale", type=float, default=0.001)
    p.add_argument("--diagnostics-interval", type=int, default=100,
                   help="First and every N optimizer steps; 0 disables diagnostics")
    p.add_argument("--precision", choices=("fp16", "fp32", "amp"), default="fp16")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--test-batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu", default=os.environ.get("GPU_ID", "0"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def make_command(args, dataset, seed):
    kd = args.kd_weight if args.kd_weight is not None else (
        200.0 if args.kd_policy == "comment" and dataset in
        {"fgvc_aircraft", "oxford_flowers", "dtd"} else 1000.0)
    suffix = "" if args.method == "baseline" else "_" + args.method
    config = ROOT / "configs/trainers/PromptKD" / (
        "vit_b16_c2_ep20_batch8_4+4ctx" + suffix + ".yaml")
    protocol = "best_val" if args.protocol == "official_best_base" else "last_step"
    epochs = 1 if args.smoke else args.epochs
    mode = "smoke" if args.smoke else "full"
    variant = (f"lr_{args.adapt_lr_mult:g}_targets_{''.join(args.lora_targets)}" if args.method == "larp"
               else f"lr_{args.adapt_lr_mult:g}_scale_{args.harp_scale:g}" if args.method == "harp" else "default")
    out = (Path(args.output_root).expanduser().resolve() / args.stage / mode / args.protocol / args.precision /
           args.method / variant / dataset / f"kd_{kd:g}" / f"rep_{args.rep_weight:g}" / f"seed_{seed}")
    cmd = [sys.executable, str(ROOT / "train.py"), "--root", str(Path(args.data_root).resolve()),
           "--seed", str(seed), "--trainer", "PromptKD",
           "--dataset-config-file", str(ROOT / "configs/datasets" / (dataset + ".yaml")),
           "--config-file", str(config), "--output-dir", str(out),
           "DATASET.NUM_SHOTS", "0", "TRAINER.MODAL", "base2novel",
           "TRAINER.PROMPTKD.KD_WEIGHT", repr(float(kd)),
           "TRAINER.PROMPTKD.TEMPERATURE", "1.0",
           "TRAINER.PROMPTKD.ADAPTATION.REP_WEIGHT", repr(float(args.rep_weight)),
           "TRAINER.PROMPTKD.ADAPTATION.LR_MULT", repr(float(args.adapt_lr_mult)),
           "TRAINER.PROMPTKD.ADAPTATION.LORA_TARGETS", repr(args.lora_targets),
           "TRAINER.PROMPTKD.ADAPTATION.HARP_SCALE", repr(float(args.harp_scale)),
           "TRAINER.PROMPTKD.DEVELOPMENT", str(args.stage == "development"),
           "TRAINER.PROMPTKD.DIAGNOSTICS.INTERVAL", str(args.diagnostics_interval),
           "TRAINER.PROMPTKD.PREC", args.precision,
           "TRAINER.PROMPTKD.REPRODUCIBLE", str(args.protocol == "fixed_last"),
           "TRAINER.PROMPTKD.WEIGHTS_ROOT", str(Path(args.weights_root).resolve()),
           "TRAINER.PROMPTKD.TEACHER_ROOT", str(Path(args.teacher_root).resolve()),
           "TEST.FINAL_MODEL", protocol, "OPTIM.MAX_EPOCH", str(epochs),
           "DATALOADER.TEST.BATCH_SIZE", str(args.test_batch_size),
           "DATALOADER.NUM_WORKERS", str(args.workers)]
    return cmd, out


def main():
    args = parser().parse_args()
    if not args.data_root:
        raise SystemExit("Set DATA_ROOT or pass --data-root.")
    # The child runs from ROOT, whereas these arguments are relative to the
    # caller's directory. Resolve before changing cwd, including custom output.
    for field in ("data_root", "weights_root", "teacher_root", "output_root"):
        setattr(args, field, str(Path(getattr(args, field)).expanduser().resolve()))
    if args.epochs < 1 or args.rep_weight < 0 or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("Invalid epochs, stability weight, or duplicate seeds.")
    if min(args.seeds) < 0 or args.test_batch_size < 1 or args.workers < 0:
        raise SystemExit("Seeds/workers must be nonnegative and batch size positive.")
    if not math.isfinite(args.rep_weight) or (args.kd_weight is not None and
            (not math.isfinite(args.kd_weight) or args.kd_weight <= 0)):
        raise SystemExit("Loss weights must be finite; KD weight must be positive.")
    if any(not math.isfinite(value) or value <= 0 for value in (args.adapt_lr_mult, args.harp_scale)):
        raise SystemExit("Adaptation LR multiplier and HARP scale must be finite and positive.")
    if args.diagnostics_interval < 0:
        raise SystemExit("Diagnostics interval must be nonnegative.")
    if args.stage == "development" and args.protocol != "fixed_last":
        raise SystemExit("Development stage uses only fixed_last, not benchmark checkpoint selection.")
    if args.lora_targets not in (["v"], ["q", "v"]):
        raise SystemExit("Controlled candidates: --lora-targets v OR --lora-targets q v.")
    if (args.method == "baseline" and args.adapt_lr_mult != 1.) or (
            args.method != "harp" and args.harp_scale != .001) or (
            args.method != "larp" and args.lora_targets != ["v"]):
        raise SystemExit("An adaptation option was supplied to a method that does not use it.")
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    if not args.dry_run:
        subprocess.run([sys.executable, str(ROOT / "tools/preflight.py"),
                        "--data-root", args.data_root, "--weights-root", args.weights_root,
                        "--teacher-root", args.teacher_root, "--datasets", *datasets],
                       cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu), check=True)
    for dataset in datasets:
        for seed in args.seeds:
            cmd, out = make_command(args, dataset, seed)
            print(shlex.join(cmd), flush=True)
            if args.dry_run:
                continue
            # A failed or differently configured run must not be resumed
            # silently by Dassl. Use a new --output-root for deliberate reruns.
            if out.exists():
                raise SystemExit(f"Output already exists: {out}. Choose a new --output-root.")
            out.mkdir(parents=True)
            sources = {}
            for relative in ("train.py", "clip/model.py", "trainers/promptkd.py",
                             "trainers/efficient_adaptation.py"):
                sources[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            record = {"command": cmd, "sources": sources, "options": vars(args),
                      "dataset": dataset, "seed": seed}
            (out / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, CUBLAS_WORKSPACE_CONFIG=":4096:8")
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
