"""Compare complete matched benchmark runs; reject mixed variants/development."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from collections import defaultdict

DATASETS = ("caltech101", "oxford_pets", "stanford_cars", "oxford_flowers",
            "food101", "fgvc_aircraft", "sun397", "dtd", "eurosat", "ucf101")
DISPLAY = ("Caltech101", "OxfordPets", "StanfordCars", "OxfordFlowers",
           "Food101", "FGVCAircraft", "SUN397", "DescribableTextures", "EuroSAT", "UCF101")
ALIASES = dict(zip(DISPLAY, DATASETS))


def read_runs(root, method, rep_weight, candidate_id=None):
    collected, specifications = {}, {}
    for path in sorted(root.rglob("metrics.json")):
        row = json.loads(path.read_text())
        if row.get("schema_version") not in (2, 3) or row.get("status") != "complete":
            continue
        if row.get("evaluation_role", "benchmark" if row["schema_version"] == 2 else None) != "benchmark":
            continue
        if row["method"] not in {"none", method}:
            continue
        if float(row["rep_weight"]) != (0.0 if row["method"] == "none" else rep_weight):
            continue
        if row.get("epochs") != 20:
            continue
        if candidate_id and row["method"] != "none":
            actual_id = hashlib.sha256(row.get("adaptation_spec", "").encode()).hexdigest()[:12]
            if candidate_id != actual_id:
                continue
        if row["schema_version"] == 3:
            specification = row.get("adaptation_spec")
            if not specification:
                raise ValueError(f"Missing adaptation specification: {path}")
            previous = specifications.setdefault(row["method"], specification)
            if previous != specification:
                raise ValueError("Mixed adaptation variants across datasets/seeds. Select ONE --candidate-id.")
        if not row.get("comparison_signature"):
            raise ValueError(f"Missing comparison identity: {path}")
        for name in ("base", "novel", "hm"):
            value = float(row[name])
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f"Invalid {name}: {path}")
        expected_hm = 2 * row["base"] * row["novel"] / (row["base"] + row["novel"]) if row["base"] + row["novel"] else 0.
        if abs(expected_hm - row["hm"]) > 1e-6:
            raise ValueError(f"Inconsistent HM: {path}")
        dataset = ALIASES.get(row["dataset"], row["dataset"])
        key = (row["method"], dataset, int(row["seed"]))
        if key in collected:
            raise ValueError(f"Duplicate method/dataset/seed: {key}. Narrow --root to ONE protocol/precision/KD policy.")
        row["source"] = str(path)
        collected[key] = row
    return collected


def compare(runs, method, seeds):
    rows, missing = [], []
    for dataset in DATASETS:
        for seed in seeds:
            base = runs.get(("none", dataset, seed))
            new = runs.get((method, dataset, seed))
            if base is None or new is None:
                missing.append(f"{dataset}/seed_{seed}")
                continue
            if base["comparison_signature"] != new["comparison_signature"]:
                raise ValueError(f"Unmatched environment/data/teacher/config: {dataset}/seed_{seed}")
            rows.append(dict(dataset=dataset, seed=seed,
                             base_baseline=base["base"], novel_baseline=base["novel"], hm_baseline=base["hm"],
                             base_candidate=new["base"], novel_candidate=new["novel"], hm_candidate=new["hm"],
                             base_delta=new["base"] - base["base"],
                             novel_delta=new["novel"] - base["novel"],
                             hm_delta=new["hm"] - base["hm"]))
    return rows, missing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("output/v3/benchmark/full/fixed_last/fp16"))
    p.add_argument("--method", required=True, choices=("harp", "larp"))
    p.add_argument("--rep-weight", type=float, default=0.)
    p.add_argument("--candidate-id", help="12-character adaptation ID printed by report_development.py")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--csv", type=Path)
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        p.error("Seeds must be distinct.")
    rows, missing = compare(read_runs(args.root, args.method, args.rep_weight, args.candidate_id), args.method, args.seeds)
    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("dataset,seeds,delta_Base,delta_Novel,delta_HM,paired_HM_std")
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    for dataset, group in groups.items():
        deltas = [row["hm_delta"] for row in group]
        means = [statistics.mean(row[name] for row in group)
                 for name in ("base_delta", "novel_delta", "hm_delta")]
        sd = statistics.stdev(deltas) if len(deltas) > 1 else float("nan")
        print(f"{dataset},{len(group)},{means[0]:+.4f},{means[1]:+.4f},{means[2]:+.4f},{sd:.4f}")
    if missing or len(args.seeds) < 3:
        print("INCOMPLETE: no effectiveness conclusion. Missing pairs:", ", ".join(missing))
        return 2
    seed_deltas = [statistics.mean(row["hm_delta"] for row in rows if row["seed"] == seed)
                   for seed in args.seeds]
    mean_hm = statistics.mean(seed_deltas)
    mean_novel = statistics.mean(row["novel_delta"] for row in rows)
    sd = statistics.stdev(seed_deltas)
    print(f"Mean paired HM delta: {mean_hm:+.4f} pp; across-seed std: {sd:.4f} pp")
    print(f"Per-seed ten-dataset HM deltas: {seed_deltas}")
    print(f"Mean Novel delta: {mean_novel:+.4f} pp")
    print("Positive datasets:", sum(statistics.mean(r["hm_delta"] for r in g) > 0
                                     for g in groups.values()), "/10")
    for method_key, base_key, novel_key, hm_key in (
        ("baseline", "base_baseline", "novel_baseline", "hm_baseline"),
        (args.method, "base_candidate", "novel_candidate", "hm_candidate")):
        b = statistics.mean(row[base_key] for row in rows)
        n = statistics.mean(row[novel_key] for row in rows)
        print(f"{method_key}: mean_B={b:.4f}, mean_N={n:.4f}, "
              f"mean_per_dataset_HM={statistics.mean(row[hm_key] for row in rows):.4f}, "
              f"HM_of_mean_BN={2*b*n/(b+n) if b+n else 0.:.4f}")
    if min(seed_deltas) > 0 and mean_novel >= 0:
        print("OBSERVED_POSITIVE_ALL_SEEDS: positive within this benchmark; not a general guarantee.")
    elif mean_hm > 0:
        print("MIXED_POSITIVE: inspect seed instability and/or Novel regressions before making claims.")
    else:
        print("NO_AVERAGE_GAIN: the proposed method does not meet the current objective.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
