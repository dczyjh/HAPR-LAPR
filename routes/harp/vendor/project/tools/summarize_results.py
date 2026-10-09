from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


ACCURACY = re.compile(r"\* accuracy:\s*([0-9.]+)%")


def parse_log(path: Path):
    split = None
    values = defaultdict(list)
    for line in path.read_text(errors="replace").splitlines():
        if "Evaluate on the *val* set" in line:
            split = "val"
        elif "Evaluate on the *test* set" in line:
            split = "test"
        match = ACCURACY.search(line)
        if match and split is not None:
            values[split].append(float(match.group(1)))
    if not values["val"] or not values["test"]:
        return None
    base = max(values["val"])
    novel = values["test"][-1]
    harmonic_mean = 2 * base * novel / (base + novel) if base + novel else 0.0
    return base, novel, harmonic_mean


def infer_run(log_path: Path, output_root: Path):
    relative = log_path.relative_to(output_root)
    # method/dataset/kd_<weight>/rep_<weight>/seed_<seed>/log.txt
    if len(relative.parts) < 6:
        return None
    method, dataset, kd_part, rep_part, seed_part = relative.parts[:5]
    if (
        not kd_part.startswith("kd_")
        or not rep_part.startswith("rep_")
        or not seed_part.startswith("seed_")
    ):
        return None
    return method, dataset, kd_part[3:], rep_part[4:], int(seed_part[5:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("output/base2new"))
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    rows = []
    for log_path in sorted(args.output_root.rglob("log.txt")):
        run = infer_run(log_path, args.output_root)
        metrics = parse_log(log_path)
        if run is None or metrics is None:
            continue
        method, dataset, kd_weight, rep_weight, seed = run
        base, novel, harmonic_mean = metrics
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "kd_weight": kd_weight,
                "rep_weight": rep_weight,
                "seed": seed,
                "base": base,
                "novel": novel,
                "hm": harmonic_mean,
            }
        )

    if not rows:
        print(f"No complete logs found under {args.output_root}")
        return 1

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote per-seed results to {args.csv}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["dataset"], row["kd_weight"], row["rep_weight"])].append(row)

    header = "method,dataset,kd_weight,rep_weight,seeds,base_mean,novel_mean,hm_mean,hm_std"
    print(header)
    for key, group in sorted(grouped.items()):
        hms = [row["hm"] for row in group]
        base_mean = statistics.mean(row["base"] for row in group)
        novel_mean = statistics.mean(row["novel"] for row in group)
        hm_mean = statistics.mean(hms)
        hm_std = statistics.stdev(hms) if len(hms) > 1 else 0.0
        method, dataset, kd_weight, rep_weight = key
        print(
            f"{method},{dataset},{kd_weight},{rep_weight},{len(group)},"
            f"{base_mean:.3f},{novel_mean:.3f},{hm_mean:.3f},{hm_std:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
