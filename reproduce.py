#!/usr/bin/env python3
"""Plan and execute the fixed baseline, main experiments, and structure ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
from runtime.execution import VARIANTS, SPECS, execution_plan, first_evaluation, read, run_queue, sha, worker


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("plan", "run"):
        q = sub.add_parser(command)
        q.add_argument("--route", choices=list(VARIANTS), required=True)
        q.add_argument("--dataset", action="append", choices=list(SPECS))
        q.add_argument("--seed", action="append", type=int, choices=[1,2,3])
        group = q.add_mutually_exclusive_group()
        group.add_argument("--variant", action="append")
        group.add_argument("--all-variants", action="store_true")
        q.add_argument("--data-root", required=True, type=Path)
        q.add_argument("--clip-root", required=True, type=Path)
        q.add_argument("--teacher-root", required=True, type=Path)
        q.add_argument("--output-root", required=True, type=Path)
        q.add_argument("--gpu", default="0", help="One CUDA_VISIBLE_DEVICES device identifier")
    q = sub.add_parser("evaluate", help="First independent evaluation of an existing completed training run")
    q.add_argument("--registration", required=True, type=Path)
    q.add_argument("--dataset", choices=list(SPECS), required=True)
    q.add_argument("--seed", type=int, choices=[1,2,3], required=True)
    q.add_argument("--variant", required=True)
    q = sub.add_parser("_worker", help=argparse.SUPPRESS)
    q.add_argument("--registration", required=True, type=Path)
    q.add_argument("--action", choices=["init","gate","train","evaluate"], required=True)
    q.add_argument("--dataset", required=True)
    q.add_argument("--seed", type=int, required=True)
    q.add_argument("--variant", required=True)
    return p


def main(argv=None):
    p = parser()
    if sys.flags.optimize:
        p.error("Do not run with -O/PYTHONOPTIMIZE: scientific assertions must remain enabled")
    args = p.parse_args(argv)
    if args.command in ("_worker", "evaluate"):
        reg = read(args.registration)
        item = {"action": "evaluate" if args.command == "evaluate" else args.action,
                "dataset": args.dataset, "seed": args.seed, "variant": args.variant}
        if item not in reg["plan"]["actions"]:
            p.error("Requested action is not registered in this run")
        if args.command == "evaluate":
            if args.registration.resolve() != Path(reg["output_root"]) / "registration.json":
                p.error("Use the original run's registration.json path")
            first_evaluation(reg, item)
        else:
            worker(reg, **item)
        return 0
    try:
        plan = execution_plan(args.route, args.dataset, args.seed, args.variant, args.all_variants)
    except ValueError as exc:
        p.error(str(exc))
    if "," in args.gpu or not args.gpu:
        p.error("Select exactly one GPU")
    reg = {"schema_version": 2, "route": args.route, "plan": plan, "gpu": args.gpu,
           "release_manifest_sha256": sha(ROOT / "metadata/release_manifest.json"),
           **{key: str(getattr(args,key).expanduser().resolve()) for key in ("data_root","clip_root","teacher_root","output_root")}}
    if args.command == "plan":
        print(json.dumps(reg, indent=2, ensure_ascii=False))
        return 0
    if not all(Path(reg[key]).is_dir() for key in ("data_root","clip_root","teacher_root")):
        p.error("Data, CLIP and teacher roots must already exist; see docs/ASSET_PREPARATION.md")
    if Path(reg["output_root"]).exists():
        p.error("Output root must be new; completed/failed runs are never overwritten")
    output = Path(reg["output_root"])
    for protected in [ROOT, *(Path(reg[key]) for key in ("data_root", "clip_root", "teacher_root"))]:
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            p.error("Output must be separate from repository, data and pretrained roots")
    result = run_queue(reg)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
