from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED_DATASETS = {
    "caltech101",
    "oxford_pets",
    "stanford_cars",
    "oxford_flowers",
    "food101",
    "fgvc_aircraft",
    "sun397",
    "dtd",
    "eurosat",
    "ucf101",
}


def bootstrap_interval(values, samples=10000, seed=0):
    if not values:
        raise ValueError("values cannot be empty")
    generator = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [generator.choice(values) for _ in values]
        means.append(statistics.mean(draw))
    means.sort()
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--method", choices=("harp", "larp"), required=True)
    parser.add_argument("--rep-weight", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    with args.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    baseline = {}
    candidate = {}
    for row in rows:
        key = (row["dataset"], row["kd_weight"], int(row["seed"]))
        metrics = tuple(float(row[name]) for name in ("base", "novel", "hm"))
        if row["method"] == "baseline" and row["rep_weight"] in {"0", "0.0"}:
            baseline[key] = metrics
        elif row["method"] == args.method and row["rep_weight"] == args.rep_weight:
            candidate[key] = metrics

    paired = sorted(set(baseline) & set(candidate))
    if not paired:
        print("No paired baseline/candidate rows found.")
        return 1

    by_dataset = defaultdict(list)
    for key in paired:
        dataset = key[0]
        deltas = tuple(c - b for c, b in zip(candidate[key], baseline[key]))
        by_dataset[dataset].append(deltas)

    print("dataset,seeds,base_delta,novel_delta,hm_delta")
    dataset_means = []
    for dataset in sorted(by_dataset):
        group = by_dataset[dataset]
        means = tuple(statistics.mean(item[index] for item in group) for index in range(3))
        dataset_means.append(means)
        print(
            f"{dataset},{len(group)},{means[0]:+.3f},{means[1]:+.3f},{means[2]:+.3f}"
        )

    base_delta = statistics.mean(item[0] for item in dataset_means)
    novel_delta = statistics.mean(item[1] for item in dataset_means)
    hm_deltas = [item[2] for item in dataset_means]
    hm_delta = statistics.mean(hm_deltas)
    ci_low, ci_high = bootstrap_interval(hm_deltas)
    positive_datasets = sum(value > 0 for value in hm_deltas)

    complete = (
        set(by_dataset) == EXPECTED_DATASETS
        and all(len(group) >= args.seeds for group in by_dataset.values())
    )
    if not complete:
        verdict = "INCOMPLETE"
    elif hm_delta > 0 and ci_low > 0 and novel_delta >= -0.1:
        verdict = "SUPPORTED"
    elif hm_delta > 0 and novel_delta >= -0.1:
        verdict = "PROMISING_BUT_UNCERTAIN"
    else:
        verdict = "NOT_SUPPORTED"

    print(
        "overall," 
        f"datasets={len(dataset_means)},base_delta={base_delta:+.3f},"
        f"novel_delta={novel_delta:+.3f},hm_delta={hm_delta:+.3f},"
        f"hm_bootstrap95=[{ci_low:+.3f},{ci_high:+.3f}],"
        f"positive_datasets={positive_datasets}/{len(dataset_means)}"
    )
    print(f"verdict={verdict}")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit("Legacy v1 tool disabled: use tools/report_results.py on structured benchmark metrics.json. "
                     "Its former bootstrap verdict was not evidence of seed-level significance.")
