#!/usr/bin/env python3
"""Recompute historical findings, without training, selection, or source edits.

The default is read-only and prints a compact preview. To create the three
deterministic artifacts, explicitly pass --output-root with a NEW directory.
Only Python's standard library is required; no network or plotting dependency.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from html import escape
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1, 2, 3)
DATASETS = ("caltech101", "oxford_pets", "stanford_cars", "oxford_flowers", "food101",
            "fgvc_aircraft", "sun397", "dtd", "eurosat", "ucf101")
LABELS = dict(zip(DATASETS, ("Caltech101", "OxfordPets", "StanfordCars", "OxfordFlowers",
                           "Food101", "FGVCAircraft", "SUN397", "DTD", "EuroSAT", "UCF101")))
HARP_METHODS = ("R0", "R1", "HARP", "mid4_d32", "last4_d16", "last4_d64")
LARP_METHODS = ("R0", "R1", "O4-r2", "O4-r1", "O4-r4", "O3-r2")
METRICS = ("base", "novel", "hm")
SOURCES = ("results/harp/complete_results.json", "results/larp/flat_results.json",
           "results/larp/r0_external_reference.json")
COMPARISONS = ("main_minus_r1", "main_minus_r0", "r1_minus_r0")


def harmonic_mean(base, novel):
    return 2.0 * base * novel / (base + novel) if base + novel else 0.0


def index_records(rows, methods, *, datasets=DATASETS, method_field="method"):
    """Reject missing/duplicate roles, even if duplicate scores are identical."""
    indexed = {}
    for raw in rows:
        dataset, seed, method = raw.get("dataset"), raw.get("seed"), raw.get(method_field)
        if dataset not in datasets or type(seed) is not int or seed not in SEEDS or method not in methods:
            raise ValueError(f"Unexpected result identity: {(dataset, seed, method)}")
        key = dataset, seed, method
        if key in indexed:
            raise ValueError(f"Duplicate result key: {key}")
        row = dict(raw)
        row["method"] = method
        for metric in METRICS:
            value = row.get(metric)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 100):
                raise ValueError(f"Invalid {metric} for {key}")
        hm = harmonic_mean(row["base"], row["novel"])
        if not math.isclose(row["hm"], hm, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"Stored HM disagrees with per-seed Base/Novel: {key}")
        row["hm"] = hm  # Recompute before seed averaging; never HM(mean B, mean N).
        indexed[key] = row
    expected = {(dataset, seed, method) for dataset in datasets for seed in SEEDS for method in methods}
    missing = expected.difference(indexed)
    if missing:
        raise ValueError(f"Missing result keys/seeds: {sorted(missing)}")
    return indexed


def arithmetic_mean(values):
    """Sum exact binary-float ratios; convert to float only once at the end."""
    values = list(values)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Mean requires nonempty finite values")
    return float(sum((Fraction(value) for value in values), Fraction(0)) / len(values))


def stats(values):
    values = list(values)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("Statistics require exactly three finite seed values")
    # statistics.stdev changed its last-bit rounding between Python versions.
    # Make our arithmetic explicit instead of rounding data or relaxing tests:
    # exact rational mean/variance of the input floats, then one float conversion
    # of variance followed by math.sqrt. The denominator is sample n-1.
    exact = [Fraction(value) for value in values]
    mean = sum(exact, Fraction(0)) / len(exact)
    variance = sum(((value - mean) ** 2 for value in exact), Fraction(0)) / (len(exact) - 1)
    return {"mean": float(mean), "sample_sd": math.sqrt(float(variance)), "values": values}


def differences(indexed, dataset, left, right):
    result = {metric: stats(indexed[dataset, seed, left][metric] - indexed[dataset, seed, right][metric]
                            for seed in SEEDS) for metric in METRICS}
    values = result["hm"]["values"]
    result["positive_hm_seed_count"] = sum(value > 0 for value in values)
    result["negative_hm_seed_count"] = sum(value < 0 for value in values)
    result["zero_hm_seed_count"] = sum(value == 0 for value in values)
    return result


def summarize_matrix(indexed, methods, main, *, datasets=DATASETS):
    per_dataset = {}
    for dataset in datasets:
        method_rows = {}
        for method in methods:
            method_rows[method] = {
                "statistics": {metric: stats(indexed[dataset, seed, method][metric] for seed in SEEDS)
                               for metric in METRICS},
                "versus": {ref: differences(indexed, dataset, method, ref) for ref in ("R1", "R0")},
            }
            vs = method_rows[method]["versus"]
            method_rows[method]["dual_positive_mean"] = vs["R1"]["hm"]["mean"] > 0 and vs["R0"]["hm"]["mean"] > 0
            method_rows[method]["dual_positive_seed_count"] = sum(
                a > 0 and b > 0 for a, b in zip(vs["R1"]["hm"]["values"], vs["R0"]["hm"]["values"]))
        comparisons = {"main_minus_r1": differences(indexed, dataset, main, "R1"),
                       "main_minus_r0": differences(indexed, dataset, main, "R0"),
                       "r1_minus_r0": differences(indexed, dataset, "R1", "R0")}
        r1, r0 = (comparisons[name]["hm"] for name in COMPARISONS[:2])
        per_dataset[dataset] = {
            "label": LABELS.get(dataset, dataset), "methods": method_rows,
            "comparisons": comparisons,
            "main_dual_positive_mean": r1["mean"] > 0 and r0["mean"] > 0,
            "main_dual_positive_seed_count": sum(a > 0 and b > 0 for a, b in zip(r1["values"], r0["values"])),
        }
    macro_methods = {}
    for method in methods:
        macro_methods[method] = {
            metric: stats(arithmetic_mean(indexed[dataset, seed, method][metric] for dataset in datasets)
                          for seed in SEEDS) for metric in METRICS}
        macro_methods[method]["versus"] = {}
        for reference in ("R1", "R0"):
            paired = {metric: stats(arithmetic_mean(
                indexed[dataset, seed, method][metric] - indexed[dataset, seed, reference][metric]
                for dataset in datasets) for seed in SEEDS) for metric in METRICS}
            paired["positive_dataset_count"] = sum(
                row["methods"][method]["versus"][reference]["hm"]["mean"] > 0 for row in per_dataset.values())
            macro_methods[method]["versus"][reference] = paired
        macro_methods[method]["dual_positive_dataset_count"] = sum(
            row["methods"][method]["dual_positive_mean"] for row in per_dataset.values())
    macro_comparisons = {}
    for name, left, right in (("main_minus_r1", main, "R1"), ("main_minus_r0", main, "R0"),
                              ("r1_minus_r0", "R1", "R0")):
        macro_comparisons[name] = {
            metric: stats(arithmetic_mean(indexed[dataset, seed, left][metric] - indexed[dataset, seed, right][metric]
                                          for dataset in datasets) for seed in SEEDS) for metric in METRICS}
        macro_comparisons[name]["positive_dataset_count"] = sum(
            row["comparisons"][name]["hm"]["mean"] > 0 for row in per_dataset.values())
        macro_comparisons[name]["positive_hm_seed_count"] = sum(
            value > 0 for value in macro_comparisons[name]["hm"]["values"])
    return {"main_method": main, "method_order": list(methods), "record_count": len(indexed),
            "datasets": per_dataset,
            "macro": {"methods": macro_methods, "comparisons": macro_comparisons,
                      "main_dual_positive_dataset_count": sum(row["main_dual_positive_mean"] for row in per_dataset.values())}}


def original_harp_index(harp, selected):
    """Use explicit historical records, not invented or reverse-engineered scores."""
    original = {key: value for key, value in selected.items() if key[2] in ("R0", "R1", "HARP")}
    history = harp["original_eurosat_seed2"]
    revisions = harp["revision_selections"]
    revised_keys = {(row["dataset"], row["seed"], row["method"]) for row in revisions}
    if len(revised_keys) != len(revisions) or revised_keys != {("eurosat", 2, "HARP"), ("eurosat", 2, "R1")}:
        raise ValueError("Unexpected HARP revision policy; explicit review is required")
    history_keys = [(row["dataset"], row["seed"], row["method"]) for row in history]
    if len(history_keys) != len(set(history_keys)) or set(history_keys) != revised_keys:
        raise ValueError("Missing or duplicate original HARP revision records")
    by_key = dict(zip(history_keys, history))
    for revision in revisions:
        key = revision["dataset"], revision["seed"], revision["method"]
        previous, current = by_key[key], selected[key]
        if previous.get("source") != revision["old_source"] or current.get("source") != revision["new_source"]:
            raise ValueError(f"Revision source does not match explicit original/selected record: {key}")
        if (not math.isclose(previous["hm"], revision["original_hm"], rel_tol=0, abs_tol=1e-8)
                or not math.isclose(current["hm"], revision["revised_hm"], rel_tol=0, abs_tol=1e-8)):
            raise ValueError(f"Revision HM mismatch: {key}")
        original[key] = dict(previous)
    return index_records(list(original.values()), ("R0", "R1", "HARP"))


def build_summary(root=ROOT):
    root = Path(root)
    documents, sources = [], []
    for rel in SOURCES:
        raw = (root / rel).read_bytes()
        documents.append(json.loads(raw))
        sources.append({"path": rel, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    harp, larp, r0 = documents
    selected = index_records(harp["records"], HARP_METHODS)
    original = original_harp_index(harp, selected)
    larp_index = index_records([dict(row, method=row["variant"]) for row in larp]
                              + [dict(row, method="R0") for row in r0["rows"]], LARP_METHODS)
    return {
        "schema_version": 1, "status": "recomputed_from_published_historical_records",
        "source_files": sources, "dataset_order": list(DATASETS), "seeds": list(SEEDS),
        "statistical_policy": {
            "hm": "For each dataset/method/seed compute 2*Base*Novel/(Base+Novel); then mean and sample SD over seeds 1,2,3.",
            "differences": "Subtract route-specific reference at matching dataset and seed before averaging; unit is percentage points (pp).",
            "macro": "Equal weight to all ten datasets. Mean equals average of dataset three-seed means; macro sample SD is over three per-seed ten-dataset macro values, NOT pooled 30 records or dataset SD.",
            "positive": "Strict >0 on unrounded values; exact zero is not positive. Seed counts are descriptive, not significance tests.",
            "dual_positive": "Dataset: mean main-R1 >0 AND mean main-R0 >0. Seed: both differences >0 at the same seed.",
            "uncertainty": "Sample SD (ddof=1), not a confidence interval. Three seeds do not establish statistical significance or guaranteed improvement.",
            "arithmetic": "Mean and sample variance are computed with exact Fraction ratios of the input binary floats. Mean is converted once to float; SD is math.sqrt(float(exact sample variance)). No decimal rounding of stored metrics; this fixes Python statistics.stdev version-dependent last-bit differences.",
        },
        "source_policy": {
            "harp": {"recorded_policy": harp["source_policy"],
                     "selected": "records: existing result-selected EuroSAT seed2 pair is retained; not relabeled as first runs.",
                     "original": "Only the two explicit original_eurosat_seed2 HARP/R1 records replace their selected counterparts; all other main records unchanged.",
                     "revision_selections": harp["revision_selections"],
                     "selection_after_results_seen": harp["selection_after_results_seen"],
                     "original_results_invalidated": harp["original_results_invalidated"],
                     "machines_equivalent_proven": harp["machines_equivalent_proven"]},
            "larp": {"main_variant": "O4-r2", "main_changed_after_ablation": False,
                     "r1": "Each LARP variant is compared with LARP's R1 at the same dataset and seed; not HARP's R1.",
                     "r0_identity": r0["identity"], "r0_note": r0["note"],
                     "same_seed_is_not_same_machine": True,
                     "same_machine_as_r1_by_variant": {
                         variant: {"true": sum(row["same_machine_as_r1"] is True for row in larp if row["variant"] == variant),
                                   "false_or_unknown": sum(row.get("same_machine_as_r1") is not True for row in larp if row["variant"] == variant)}
                         for variant in LARP_METHODS if variant != "R0"},
                     "replays": "The separate Pets diagnostic replay is not an extra seed and is not substituted into flat_results.",
                     "ablation_limits": "Rank/scale and layer/capacity coupling and cross-machine reference reuse are not pure causal contrasts."},
            "cross_route": "HARP and LARP are independent methods with route-specific controls. Do not rank methods across routes from equal-looking reference numbers or unmatched environments.",
            "scope": "Recalculation of lightweight historical records only; no training, new evaluation, new result selection or full checkpoint/log audit. No per-epoch Novel values inferred.",
        },
        "routes": {
            "harp": {"selected": summarize_matrix(selected, HARP_METHODS, "HARP"),
                     "original": summarize_matrix(original, ("R0", "R1", "HARP"), "HARP")},
            "larp": {"recorded": summarize_matrix(larp_index, LARP_METHODS, "O4-r2")},
        },
    }


def panel_items(summary):
    return (("HARP 选用口径", summary["routes"]["harp"]["selected"]),
            ("HARP 原始主序列", summary["routes"]["harp"]["original"]),
            ("LARP O4-r2", summary["routes"]["larp"]["recorded"]))


def pm(stat):
    return f'{stat["mean"]:.6f} ± {stat["sample_sd"]:.6f}'


def markdown(summary):
    lines = ["# 可重算的历史主结果", "",
             "本页由 `tools/build_findings.py` 从包内原始结果 JSON 生成；不是本次重新训练或评估。", "",
             "先逐 seed 计算 HM，再取三 seed 均值及样本标准差（±，ddof=1）。差值为同 dataset/seed 相减后统计，单位 pp；“正 seed”是严格大于零的个数，不是显著性证明。宏平均对十个数据集等权；宏 SD 是三个 seed 的十集宏均值之间的样本 SD，不是数据集间 SD。", "",
             "HARP 与 LARP 是独立路线，下表不能用于跨路线直接排名。LARP 的 R0 是外部历史参照，不是同期同机控制。", "",
             "## 主结果宏平均", "",
             "| 口径 | R0 HM | R1 HM | 主方法 HM | 主−R1 (pp) | 主−R0 (pp) | R1−R0 (pp) | 双正数据集 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for title, panel in panel_items(summary):
        macro = panel["macro"]
        cells = [title] + [pm(macro["methods"][method]["hm"]) for method in ("R0", "R1", panel["main_method"])]
        cells += [pm(macro["comparisons"][name]["hm"]) for name in COMPARISONS]
        cells += [f'{macro["main_dual_positive_dataset_count"]}/10']
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "双正数据集：三 seed 平均 HM 同时超过各自 R1 和 R0；不是同时超过两个参照的 seed 总数。", "",
              "## HARP 原始／选用敏感性", "",
              "现有选用口径使用看过结果后指定的 EuroSAT seed2 配对复放；原始记录未作废。以下两条替换被明确记录，不能只展示选用均值并声称全部首次运行稳定。", "",
              "| 数据集 / seed / 方法 | 原始 HM | 选用 HM | 差 (pp) |", "| --- | --- | --- | --- |"]
    for row in summary["source_policy"]["harp"]["revision_selections"]:
        lines.append(f'| {row["dataset"]} / {row["seed"]} / {row["method"]} | {row["original_hm"]:.6f} | {row["revised_hm"]:.6f} | {row["revised_hm"] - row["original_hm"]:+.6f} |')
    for title, panel in panel_items(summary):
        lines += ["", f"## {title}：全十集", "",
                  "| 数据集 | 主方法 HM | 主−R1 (pp) | 主−R0 (pp) | R1−R0 (pp) | 正 seed 主/R1 | 正 seed 主/R0 | 同 seed 双正 | 均值双正 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for dataset in DATASETS:
            row = panel["datasets"][dataset]
            comparisons = row["comparisons"]
            cells = [row["label"], pm(row["methods"][panel["main_method"]]["statistics"]["hm"])]
            cells += [pm(comparisons[name]["hm"]) for name in COMPARISONS]
            cells += [f'{comparisons[name]["positive_hm_seed_count"]}/3' for name in COMPARISONS[:2]]
            cells += [f'{row["main_dual_positive_seed_count"]}/3', "是" if row["main_dual_positive_mean"] else "否"]
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## 固定消融的宏 HM（不事后更换主方法）", "",
              "| 路线 / 口径 | 变体 | 宏 HM | 对本路线 R1 (pp) | 对本路线 R0 (pp) | 双正数据集 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for title, panel in (panel_items(summary)[0], panel_items(summary)[2]):
        for method in panel["method_order"]:
            if method in ("R0", "R1"):
                continue
            means = panel["macro"]["methods"]
            lines.append(f'| {title} | {method} | {pm(means[method]["hm"])} | {pm(means[method]["versus"]["R1"]["hm"])} | {pm(means[method]["versus"]["R0"]["hm"])} | {means[method]["dual_positive_dataset_count"]}/10 |')
    lines += ["", "## 来源与边界", "",
              "完整逐 seed 数值、Base/Novel、全部变体统计、配对正负计数、原／选用策略和来源 SHA 见 [summary.json](summary.json)。图中的点为三 seed 平均差（不是逐 seed 极值或置信区间），全部正负值按线性轴展示；样本 SD 见表。没有推造逐轮 Novel 轨迹。", "",
              "HARP 跨机等价未经证明；LARP 部分消融复用跨机参照，且 rank 与缩放、层数与容量存在耦合。既有 negative results、Pets 复放及 HARP 原始崩落不删除；本汇总不把复放算成第四 seed。", "",
              "以下 SHA 指发布包中实际读取的副本；历史上游来源及路径转换说明见 `metadata/publication_sources.json`。", "",
              "| 来源（相对项目根） | Bytes | SHA-256 |", "| --- | --- | --- |"]
    for source in summary["source_files"]:
        lines.append(f'| `{source["path"]}` | {source["bytes"]} | `{source["sha256"]}` |')
    return "\n".join(lines) + "\n"


def svg(summary):
    """Two independent, full-range linear panels; labels retain near-zero signs."""
    width, height = 1700, 1120
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             '<title id="title">Historical three-seed mean HM differences, independent HARP and LARP panels</title>',
             '<desc id="desc">All ten datasets and all signed mean differences versus route-specific R1 and R0; HARP original and selected policies are shown together. Unit percentage points. No route ranking, truncation, or inferred Novel trajectories.</desc>',
             '<rect width="1700" height="1120" fill="#ffffff"/>',
             '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172b3a}.title{font-size:27px;font-weight:700}.heading{font-size:22px;font-weight:700}.body{font-size:15px}.small{font-size:13px}.value{font-size:12px;font-variant-numeric:tabular-nums}</style>']

    def text(x, y, content, css="body", anchor="start", color=None):
        attr = f' style="fill:{color}"' if color else ""
        parts.append(f'<text x="{x:.2f}" y="{y:.2f}" class="{css}" text-anchor="{anchor}"{attr}>{escape(str(content))}</text>')

    text(40, 44, "Historical HM differences | three-seed means", "title")
    text(40, 73, "Independent methods and route-specific controls; descriptive comparisons, not cross-route rankings.")
    selected = summary["routes"]["harp"]["selected"]
    original = summary["routes"]["harp"]["original"]
    recorded = summary["routes"]["larp"]["recorded"]
    configs = [(40, "HARP | selected + original sensitivity", selected,
                [("Selected − R1", selected, COMPARISONS[0], "#00769b", False),
                 ("Selected − R0", selected, COMPARISONS[1], "#bc5000", False),
                 ("R1 − R0", selected, COMPARISONS[2], "#626c75", False),
                 ("Original − R1", original, COMPARISONS[0], "#00769b", True),
                 ("Original − R0", original, COMPARISONS[1], "#bc5000", True)]),
               (880, "LARP | fixed main O4-r2", recorded,
                [("O4-r2 − R1", recorded, COMPARISONS[0], "#00769b", False),
                 ("O4-r2 − R0", recorded, COMPARISONS[1], "#bc5000", False),
                 ("R1 − R0", recorded, COMPARISONS[2], "#626c75", False)])]
    for origin, title, panel, series in configs:
        text(origin, 120, title, "heading")
        subtitle = ("Filled = main/control; hollow = HARP original. Values in pp."
                    if any(hollow for _, _, _, _, hollow in series)
                    else "Only filled markers: main/control. Values in pp.")
        text(origin, 146, subtitle, "small")
        for j, (label, _, _, color, hollow) in enumerate(series):
            lx, ly = origin + (j % 3) * 245, 177 + (j // 3) * 23
            parts.append(f'<circle cx="{lx + 5}" cy="{ly - 5}" r="4" fill="{"white" if hollow else color}" stroke="{color}" stroke-width="2"/>')
            text(lx + 16, ly, label, "small")
        values = [p["datasets"][d]["comparisons"][key]["hm"]["mean"] for _, p, key, _, _ in series for d in DATASETS]
        span = max(values + [0]) - min(values + [0])
        padding = max(span * 0.07, 0.25)
        lower, upper = min(values + [0]) - padding, max(values + [0]) + padding
        chart_left, chart_right = origin + 142, origin + 686
        chart_top, chart_bottom = 238, 988

        def x(value):
            return chart_left + (value - lower) / (upper - lower) * (chart_right - chart_left)

        for tick in range(math.ceil(lower), math.floor(upper) + 1):
            xx = x(tick)
            parts.append(f'<line x1="{xx:.2f}" x2="{xx:.2f}" y1="{chart_top}" y2="{chart_bottom}" stroke="{"#758695" if tick == 0 else "#e4e9ed"}" stroke-width="{1.7 if tick == 0 else 1}"/>')
            text(xx, chart_bottom + 23, f"{tick:+d}" if tick else "0", "small", "middle")
        for i, dataset in enumerate(DATASETS):
            ycenter = chart_top + 38 + i * 74
            if i % 2 == 0:
                parts.append(f'<rect x="{origin}" y="{ycenter - 34}" width="{chart_right - origin + 112}" height="68" fill="#e9eff3" opacity="0.32"/>')
            text(origin, ycenter + 5, LABELS[dataset], "small")
            for j, (_, p, key, color, hollow) in enumerate(series):
                value = p["datasets"][dataset]["comparisons"][key]["hm"]["mean"]
                yy = ycenter + (j - (len(series) - 1) / 2) * 13
                xx = x(value)
                parts.append(f'<line x1="{x(0):.2f}" x2="{xx:.2f}" y1="{yy:.2f}" y2="{yy:.2f}" stroke="{color}" stroke-opacity="0.38"/>')
                parts.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="3.5" fill="{"white" if hollow else color}" stroke="{color}" stroke-width="1.7"/>')
                # Full small values remain numerically explicit; no zoomed near-zero inset.
                label = f"{value:+.6f}" if 0 < abs(value) < 0.01 else f"{value:+.3f}"
                text(xx + (8 if value >= 0 else -8), yy + 4, label, "value", "start" if value >= 0 else "end", color)
        text((chart_left + chart_right) / 2, 1032, "Mean ΔHM (percentage points; independent linear axis)", "small", "middle")
    text(40, 1073, "HARP: post-result EuroSAT seed2 selection is shown beside the still-valid original records. LARP R0 is an external historical reference.", "small")
    text(40, 1098, "No mean is clipped. Per-seed values and sample SD, including original failures/instability, are in main_comparison.md and summary.json.", "small")
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_findings(summary, output_root):
    """Never overwrite an existing directory or any existing artifact."""
    payloads = {"summary.json": json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                "main_comparison.md": markdown(summary), "main_hm_deltas.svg": svg(summary)}
    target = Path(output_root).expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing existing output directory: {target}")
    target.mkdir(parents=True, exist_ok=False)
    for filename, payload in payloads.items():
        with (target / filename).open("x", encoding="utf-8") as handle:
            handle.write(payload)
    return list(payloads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, help="Explicit NEW output directory; omission is read-only")
    args = parser.parse_args(argv)
    try:
        summary = build_summary()
        files = write_findings(summary, args.output_root) if args.output_root is not None else []
        print(json.dumps({"status": summary["status"], "mode": "write_new_directory" if files else "read_only",
                          "written_files": files, "source_files": summary["source_files"],
                          "main_macro_preview": {title: panel["macro"] for title, panel in panel_items(summary)}},
                         ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError) as exc:
        print(f"Findings not generated: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
