"""Compare candidates on Base development only; never read test accuracies."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

try:
    from .report_results import DATASETS, ALIASES
except ImportError:
    from report_results import DATASETS, ALIASES


def read_development(root, epochs):
    rows = {}
    for path in sorted(root.rglob("development_metrics.json")):
        row = json.loads(path.read_text())
        if row.get("schema_version") != 3 or row.get("status") != "complete":
            continue
        if row.get("evaluation_role") != "development_base" or any(k in row for k in ("base", "novel", "hm")):
            raise ValueError(f"Not an isolated development result: {path}")
        if row.get("epochs") != epochs:
            continue
        score = row["base_development"]
        if not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError(f"Invalid development accuracy: {path}")
        if not row.get("adaptation_spec") or not row.get("comparison_signature"):
            raise ValueError(f"Missing identity: {path}")
        variant = hashlib.sha256(row["adaptation_spec"].encode()).hexdigest()[:12]
        dataset = ALIASES.get(row["dataset"], row["dataset"])
        key = (row["method"], variant, dataset, row["seed"])
        if key in rows:
            raise ValueError(f"Duplicate development result {key}; narrow --root.")
        rows[key] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/v3/development/full/fixed_last/fp16"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Datasets and seeds must be unique.")
    runs = read_development(args.root, args.epochs)
    baselines = {}
    candidates = defaultdict(dict)
    for (method, variant, dataset, seed), row in runs.items():
        if dataset not in args.datasets or seed not in args.seeds:
            continue
        if method == "none":
            if float(row["rep_weight"]) != 0.:
                continue
            if (dataset, seed) in baselines:
                raise ValueError("Multiple baseline configurations; narrow --root.")
            baselines[(dataset, seed)] = row
        else:
            candidates[(method, variant)][(dataset, seed)] = row
    if not candidates:
        print("No matching development candidates; no conclusion.")
        return 2
    complete = True
    for (method, variant), group in sorted(candidates.items()):
        print(f"\n{method}/{variant}\n{next(iter(group.values()))['adaptation_spec']}")
        deltas, missing = [], []
        for dataset in args.datasets:
            for seed in args.seeds:
                key = (dataset, seed)
                baseline, candidate = baselines.get(key), group.get(key)
                if baseline is None or candidate is None:
                    missing.append(f"{dataset}/{seed}")
                    continue
                if candidate["comparison_signature"] != baseline["comparison_signature"]:
                    raise ValueError(f"Unmatched development conditions: {key}")
                delta = candidate["base_development"] - baseline["base_development"]
                print(f"{dataset}/seed_{seed}: Base-dev delta={delta:+.4f} pp")
                deltas.append(delta)
        if missing:
            complete = False
            print("INCOMPLETE; do not rank incomplete candidates. Missing:", ", ".join(missing))
        else:
            print(f"Matched mean Base-dev delta={statistics.mean(deltas):+.4f} pp")
    print("\nDevelopment selection evidence only. No Novel/HM claim; freeze the candidate before benchmark evaluation.")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
