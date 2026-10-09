#!/usr/bin/env python3
"""Summarize newly passed formal runs without training, re-evaluation or selection.

Example:
  python tools/summarize_run.py --run-root /runs/harp --run-root /runs/r0

Only formal/<variant>/audited_metrics.json is read as a result; the identical
evaluation copy and pre-hard-check metrics are deliberately excluded. Three-
seed statistics require all of seeds 1, 2, 3. JSON output never overwrites an
existing file. This tool trusts recorded passed audits; it does not rerun them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

SEEDS = (1, 2, 3)
METRICS = ("base", "novel", "hm")
VARIANTS = {"harp": ("r1", "harp", "mid4_d32", "last4_d16", "last4_d64"),
            "larp": ("r1", "o4-r2", "o4-r1", "o4-r4", "o3-r2"), "r0": ("r0",)}
DATASETS = ("caltech101", "oxford_pets", "stanford_cars", "oxford_flowers", "food101",
            "fgvc_aircraft", "sun397", "dtd", "eurosat", "ucf101")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _key_text(key):
    return "/".join(map(str, key))


def _path_key(path):
    # <run>/<route>/<dataset>/seed_N/formal/<variant>/audited_metrics.json
    if len(path.parents) < 5 or path.parent.parent.name != "formal":
        raise ValueError(f"Unexpected formal result layout: {path}")
    route, dataset, seed_dir, variant = (path.parents[4].name, path.parents[3].name,
                                          path.parents[2].name, path.parent.name)
    if route not in VARIANTS or dataset not in DATASETS or variant not in VARIANTS[route]:
        raise ValueError(f"Unknown route/dataset/variant in result path: {path}")
    if seed_dir not in ("seed_1", "seed_2", "seed_3"):
        raise ValueError(f"Unexpected seed directory: {path}")
    return route, dataset, int(seed_dir[-1]), variant


def _validate_passed(data, key, path):
    route, dataset, seed, variant = key
    if tuple(data.get(name) for name in ("route", "dataset", "seed", "variant")) != key:
        raise ValueError(f"Result identity differs from its directory: {path}")
    if type(data["seed"]) is not int or data.get("epochs") != 20:
        raise ValueError(f"Expected a 20-epoch formal seed 1/2/3 run: {path}")
    if type(data.get("selected_epoch")) is not int or not 1 <= data["selected_epoch"] <= 20:
        raise ValueError(f"Invalid best epoch: {path}")
    if data.get("canonical_evaluation") != "first_independent_reload":
        raise ValueError(f"Not the first independently reloaded evaluation: {path}")
    for metric in METRICS:
        value = data.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Invalid {metric}: {path}")
    expected_hm = 2 * data["base"] * data["novel"] / (data["base"] + data["novel"]) if data["base"] + data["novel"] else 0.
    if not math.isclose(data["hm"], expected_hm, rel_tol=0., abs_tol=1e-8):
        raise ValueError(f"HM is inconsistent with this run's Base/Novel: {path}")
    expected_kd = 200 if dataset in ("dtd", "fgvc_aircraft", "oxford_flowers") else 1000
    if data.get("kd") != expected_kd:
        raise ValueError(f"KD weight differs from the fixed dataset protocol: {path}")
    for field in ("checkpoint_sha256", "release_manifest_sha256"):
        if not isinstance(data.get(field), str) or not HEX64.fullmatch(data[field]):
            raise ValueError(f"Missing or invalid {field}: {path}")
    best_last = data.get("best_last_sha256")
    if (not isinstance(best_last, list) or len(best_last) != 2
            or any(not isinstance(value, str) or not HEX64.fullmatch(value) for value in best_last)
            or best_last[0] != data["checkpoint_sha256"]):
        raise ValueError(f"Invalid best/last checkpoint audit hashes: {path}")
    checks = data.get("reload_audit", {})
    for name in ("native_val", "native_test", "selection_val"):
        check = checks.get(name, {})
        if (check.get("row_scaled_failed_elements") != 0
                or check.get("hard_metric") != "symmetric_per_example_infinity_norm"):
            raise ValueError(f"Missing or failed registered reload hard check {name}: {path}")
    counts = data.get("counts_base_novel")
    if not isinstance(counts, list) or len(counts) != 2:
        raise ValueError(f"Missing Base/Novel evaluation counts: {path}")
    for metric, pair in zip(("base", "novel"), counts):
        if (not isinstance(pair, list) or len(pair) != 2 or any(type(x) is not int for x in pair)
                or not 0 <= pair[1] <= pair[0] or pair[0] <= 0
                or not math.isclose(data[metric], 100 * pair[1] / pair[0], rel_tol=0., abs_tol=1e-8)):
            raise ValueError(f"{metric} disagrees with evaluator counts: {path}")
    return {"route": route, "dataset": dataset, "seed": seed, "variant": variant,
            "status": "passed", "kd": expected_kd, **{name: float(data[name]) for name in METRICS},
            "selected_epoch": data["selected_epoch"], "epochs": 20,
            "counts_base_novel": counts, "checkpoint_sha256": data["checkpoint_sha256"],
            "release_manifest_sha256": data["release_manifest_sha256"],
            "source_file": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _registered_groups(root):
    path = root / "registration.json"
    if not path.is_file():
        return {}
    data = _read(path)
    route, plan = data.get("route"), data.get("plan", {})
    if route not in VARIANTS or plan.get("route") != route:
        raise ValueError(f"Unknown or inconsistent registration route: {path}")
    datasets, variants, seeds = (plan.get(name, []) for name in ("datasets", "variants", "seeds"))
    if (not datasets or not variants or not seeds or any(x not in DATASETS for x in datasets)
            or any(x not in VARIANTS[route] for x in variants) or any(type(x) is not int or x not in SEEDS for x in seeds)):
        raise ValueError(f"Invalid registered dataset/variant/seed scope: {path}")
    return {(route, dataset, variant): set(seeds) for dataset in datasets for variant in variants}


def _reference_key(row, reference):
    if reference == "r0":
        return "r0", row["dataset"], row["seed"], "r0"
    if row["route"] == "r0":
        return None  # There is no single R1 shared by the two routes.
    return row["route"], row["dataset"], row["seed"], "r1"


def _compare(row, reference, passed):
    key = _reference_key(row, reference)
    if key is None:
        return {"status": "not_applicable", "reason": "R1 is route-specific; R0 has no unique R1 reference", "deltas_pp": None}
    target = passed.get(key)
    if target is None:
        return {"status": "missing_reference", "reference_key": list(key), "deltas_pp": None}
    if target["release_manifest_sha256"] != row["release_manifest_sha256"]:
        return {"status": "incompatible_release", "reference_key": list(key),
                "reason": "Release manifests differ; do not silently combine code revisions", "deltas_pp": None,
                "reference_source_file": target["source_file"]}
    return {"status": "paired", "reference_key": list(key),
            "deltas_pp": {name: row[name] - target[name] for name in METRICS},
            "reference_source_file": target["source_file"], "reference_source_sha256": target["source_sha256"]}


def _stats(rows):
    if len(rows) != 3:
        raise ValueError("Three-seed statistics must have exactly three entries")
    return {name: {"mean": statistics.mean(row[name] for row in rows),
                   "sample_sd": statistics.stdev(row[name] for row in rows)} for name in METRICS}


def summarize(run_roots):
    roots = [Path(value).expanduser().resolve() for value in run_roots]
    if not roots or any(not root.is_dir() for root in roots):
        raise ValueError("Every --run-root must be an existing directory")
    seen, passed, excluded, planned, warnings = {}, {}, [], {}, []
    for root in roots:
        for key, seeds in _registered_groups(root).items():
            planned.setdefault(key, set()).update(seeds)
        paths = sorted(root.rglob("audited_metrics.json"))
        for path in paths:
            if path.parent.parent.name != "formal":
                continue
            if path.is_symlink():
                raise ValueError(f"Result symlink is not an independent formal record: {path}")
            key = _path_key(path)
            if key in seen:
                raise ValueError(f"Duplicate result key {_key_text(key)}: {seen[key]} and {path}; no selection is allowed")
            seen[key] = str(path)
            data = _read(path)
            group = key[0], key[1], key[3]
            planned.setdefault(group, set())
            if data.get("status") != "passed":
                excluded.append({"key": list(key), "source_file": str(path), "status": data.get("status", "missing_status"),
                                 "reason": "Only passed formal audited metrics enter the main statistics"})
                continue
            passed[key] = _validate_passed(data, key, path)
    records = [passed[key] for key in sorted(passed)]
    for row in records:
        row["comparisons"] = {name: _compare(row, name, passed) for name in ("r1", "r0")}
    groups = []
    for key in sorted(planned):
        route, dataset, variant = key
        rows = [passed[(route, dataset, seed, variant)] for seed in SEEDS if (route, dataset, seed, variant) in passed]
        missing = [seed for seed in SEEDS if (route, dataset, seed, variant) not in passed]
        manifests = sorted({row["release_manifest_sha256"] for row in rows})
        complete = not missing and len(manifests) == 1
        group = {"route": route, "dataset": dataset, "variant": variant,
                 "passed_seeds": [row["seed"] for row in rows], "missing_seeds": missing,
                 "registered_seeds": sorted(planned[key]), "release_manifest_sha256": manifests,
                 "status": "complete_three_seed" if complete else "incomplete" if missing else "incompatible_release",
                 "three_seed": _stats(rows) if complete else None, "comparisons": {}}
        for reference in ("r1", "r0"):
            comparisons = [row["comparisons"][reference] for row in rows]
            missing_refs = [row["seed"] for row in rows if row["comparisons"][reference]["status"] == "missing_reference"]
            incompatible = [row["seed"] for row in rows if row["comparisons"][reference]["status"] == "incompatible_release"]
            paired = complete and all(value["status"] == "paired" for value in comparisons)
            status = ("paired_three_seed" if paired else "not_applicable" if route == "r0" and reference == "r1"
                      else "incomplete_target" if missing else "incompatible_release" if incompatible or len(manifests) > 1
                      else "missing_reference")
            group["comparisons"][reference] = {"status": status, "missing_reference_seeds": missing_refs,
                                               "incompatible_release_seeds": incompatible,
                                               "three_seed_deltas_pp": _stats([c["deltas_pp"] for c in comparisons]) if paired else None}
        groups.append(group)
    if not records:
        warnings.append("No passed formal results found; no scores or three-seed statistics were generated.")
    if any(group["missing_seeds"] for group in groups):
        warnings.append("Incomplete groups have no three-seed mean/SD; missing or failed seeds are not interpolated.")
    if any(len(group["release_manifest_sha256"]) > 1 for group in groups):
        warnings.append("Different release manifests occur within a group; no cross-revision mean/SD is reported.")
    return {"schema_version": 1, "run_roots": [str(root) for root in roots], "required_seeds": list(SEEDS),
            "counts": {"passed_formal": len(records), "excluded_nonpassed": len(excluded), "groups": len(groups),
                       "complete_three_seed_groups": sum(g["status"] == "complete_three_seed" for g in groups)},
            "statistics": {"dispersion": "sample standard deviation, ddof=1", "delta_units": "percentage points",
                           "hm_aggregation": "mean/SD of the three per-seed HMs, not HM of mean Base/Novel",
                           "reference_pairing": "same dataset/seed/release; R1 must be from the same route; R0 is an independent route",
                           "selection": "none; duplicate keys are fatal; no best-run/seed selection",
                           "audit_scope": "reads passed formal audit records; does not repeat checkpoint/GPU audits",
                           "cross_route_caveat": "Matching keys do not imply identical backends, hardware or drivers across routes"},
            "records": records, "groups": groups, "excluded": excluded, "warnings": warnings}


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _delta_cell(comparison, aggregate=False):
    key = "three_seed_deltas_pp" if aggregate else "deltas_pp"
    values = comparison.get(key)
    if values is None:
        return {"not_applicable": "不适用", "missing_reference": "缺基线", "incompatible_release": "版本不匹配",
                "incomplete_target": "目标缺 seed"}.get(comparison["status"], comparison["status"])
    if aggregate:
        return " / ".join(f"{values[name]['mean']:+.4f} ± {values[name]['sample_sd']:.4f}" for name in METRICS)
    return " / ".join(f"{values[name]:+.4f}" for name in METRICS)


def markdown(report):
    lines = ["# 新运行结果汇总", "", "仅包含正式核验 `passed` 的首次独立重载成绩；增幅为百分点（pp），顺序均为 Base / Novel / HM。",
             "不同路线的 R1 不混用；跨路线对比不意味着后端、硬件或驱动相同。", "", "## 单 seed", "",
             "| 路线 | 数据集 | seed | 方法 | KD | Base (%) | Novel (%) | HM (%) | 最佳 epoch | Δ 对 R1 (pp) | Δ 对 R0 (pp) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["records"]:
        values = [row["route"], row["dataset"], row["seed"], row["variant"], row["kd"],
                  *(f"{row[name]:.4f}" for name in METRICS), row["selected_epoch"],
                  _delta_cell(row["comparisons"]["r1"]), _delta_cell(row["comparisons"]["r0"])]
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    lines += ["", "## 三 seed 均值 ± 样本标准差", "", "只有 seed 1、2、3 全部通过且版本一致才计算；HM 先逐 seed 计算再取均值。",
              "增幅的标准差是三个配对差值的样本标准差，不是两组标准差相减。", "",
              "| 路线 | 数据集 | 方法 | seed 完整性 | Base (%) | Novel (%) | HM (%) | Δ 对 R1 (pp) | Δ 对 R0 (pp) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for group in report["groups"]:
        stats = group["three_seed"]
        completeness = "1,2,3" if stats is not None else "缺 " + ",".join(map(str, group["missing_seeds"])) if group["missing_seeds"] else "版本不匹配"
        values = [group["route"], group["dataset"], group["variant"], completeness,
                  *(f"{stats[name]['mean']:.4f} ± {stats[name]['sample_sd']:.4f}" if stats else "—" for name in METRICS),
                  _delta_cell(group["comparisons"]["r1"], True), _delta_cell(group["comparisons"]["r0"], True)]
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    lines += ["", "## 缺项与核验范围", "", f"通过正式结果 {report['counts']['passed_formal']} 条；未通过而排除 {report['counts']['excluded_nonpassed']} 条。"]
    for group in report["groups"]:
        label = _key_text((group["route"], group["dataset"], group["variant"]))
        if group["missing_seeds"]:
            lines.append(f"- {_cell(label)}：缺 passed seed {','.join(map(str, group['missing_seeds']))}。")
        for reference, comparison in group["comparisons"].items():
            if comparison["missing_reference_seeds"]:
                lines.append(f"- {_cell(label)}：{reference.upper()} 缺 seed {','.join(map(str, comparison['missing_reference_seeds']))}。")
            if comparison["incompatible_release_seeds"]:
                lines.append(f"- {_cell(label)}：与 {reference.upper()} 的版本不匹配，seed {','.join(map(str, comparison['incompatible_release_seeds']))}。")
    for excluded in report["excluded"]:
        lines.append(f"- 排除 {_cell(_key_text(excluded['key']))}：状态 `{_cell(excluded['status'])}`。")
    for warning in report["warnings"]:
        lines.append(f"- {_cell(warning)}")
    lines += ["", "本工具只读取现有 passed 审计记录，不新增评估、不验证新的 GPU 等价性、不覆盖历史成绩，也不作显著性或效果保证。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", required=True, type=Path,
                        help="New execution output root; repeat for independent baseline/HARP/LARP runs")
    parser.add_argument("--output", type=Path, help="Optionally save a new JSON report; an existing file is never overwritten")
    args = parser.parse_args(argv)
    try:
        report = summarize(args.run_root)
        if args.output is not None:
            output = args.output.expanduser().resolve()
            historical = Path(__file__).resolve().parents[1] / "results"
            if output.is_relative_to(historical):
                raise ValueError("Do not write a new-run report into the historical results directory")
            # No mkdir or overwrite of user directories/files as a side effect.
            with output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        print(markdown(report), end="")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
