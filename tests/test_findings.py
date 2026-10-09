"""Offline checks for historical findings; no model, network, or training imports."""
from contextlib import redirect_stdout
import copy
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_findings", ROOT / "tools/build_findings.py")
findings = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(findings)


def fixture_rows():
    rows = []
    for seed, (base, novel) in enumerate(((100, 10), (10, 100), (50, 50)), 1):
        for method, b, n in (("R0", 40, 40), ("R1", 45, 45), ("M", base, novel)):
            rows.append({"dataset": "example", "seed": seed, "method": method,
                         "base": b, "novel": n, "hm": findings.harmonic_mean(b, n)})
    return rows


class FindingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = findings.build_summary()

    def fixture_panel(self, rows=None):
        methods = ("R0", "R1", "M")
        index = findings.index_records(fixture_rows() if rows is None else rows,
                                       methods, datasets=("example",))
        return findings.summarize_matrix(index, methods, "M", datasets=("example",))

    def test_missing_seed_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing result keys/seeds"):
            self.fixture_panel(fixture_rows()[:-1])

    def test_duplicate_key_rejected_even_for_identical_score(self):
        rows = fixture_rows()
        with self.assertRaisesRegex(ValueError, "Duplicate result key"):
            self.fixture_panel(rows + [copy.deepcopy(rows[0])])

    def test_unknown_seed_and_nonfinite_rejected(self):
        rows = fixture_rows()
        rows[0]["seed"] = 4
        with self.assertRaisesRegex(ValueError, "Unexpected result identity"):
            self.fixture_panel(rows)
        rows = fixture_rows()
        rows[0]["base"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid base"):
            self.fixture_panel(rows)

    def test_inconsistent_stored_hm_rejected(self):
        rows = fixture_rows()
        rows[0]["hm"] += 0.001
        with self.assertRaisesRegex(ValueError, "Stored HM disagrees"):
            self.fixture_panel(rows)

    def test_hm_before_seed_mean_and_sample_sd(self):
        row = self.fixture_panel()["datasets"]["example"]["methods"]["M"]["statistics"]
        values = [2000 / 110, 2000 / 110, 50.0]
        self.assertAlmostEqual(row["hm"]["mean"], statistics.mean(values), places=12)
        self.assertAlmostEqual(row["hm"]["sample_sd"], statistics.stdev(values), places=12)
        self.assertNotAlmostEqual(row["hm"]["mean"], findings.harmonic_mean(row["base"]["mean"], row["novel"]["mean"]), places=6)
        self.assertNotAlmostEqual(row["hm"]["sample_sd"], statistics.pstdev(values), places=6)

    def test_explicit_fraction_arithmetic_is_order_independent(self):
        values = [0.1, 0.2, 1000.3]
        exact = [Fraction.from_float(value) for value in values]
        mean = sum(exact, Fraction(0)) / 3
        variance = sum(((value - mean) ** 2 for value in exact), Fraction(0)) / 2
        expected_sd = math.sqrt(float(variance))
        for order in (values, list(reversed(values)), values[1:] + values[:1]):
            actual = findings.stats(order)
            self.assertEqual(actual["mean"].hex(), float(mean).hex())
            self.assertEqual(actual["sample_sd"].hex(), expected_sd.hex())
            self.assertEqual(findings.arithmetic_mean(order).hex(), float(mean).hex())
        # Exact summation also preserves a small residual around cancellation.
        self.assertEqual(findings.arithmetic_mean([1e16, 1.0, -1e16]), 1.0 / 3.0)

    def test_controls_subtracted_within_seed_and_sign_counts(self):
        row = self.fixture_panel()["datasets"]["example"]
        self.assertEqual(row["comparisons"]["r1_minus_r0"]["hm"]["values"], [5.0, 5.0, 5.0])
        expected = [2000 / 110 - 45, 2000 / 110 - 45, 5.0]
        self.assertEqual(row["comparisons"]["main_minus_r1"]["hm"]["values"], expected)
        self.assertEqual(row["comparisons"]["main_minus_r1"]["positive_hm_seed_count"], 1)
        self.assertEqual(row["main_dual_positive_seed_count"], 1)
        self.assertFalse(row["main_dual_positive_mean"])

    def test_zero_not_counted_as_positive(self):
        rows = fixture_rows()
        for row in rows:
            if row["method"] == "M":
                row.update(base=45, novel=45, hm=45)
        row = self.fixture_panel(rows)["datasets"]["example"]
        comparison = row["comparisons"]["main_minus_r1"]
        self.assertEqual(comparison["positive_hm_seed_count"], 0)
        self.assertEqual(comparison["zero_hm_seed_count"], 3)
        self.assertFalse(row["main_dual_positive_mean"])

    def test_real_harp_matches_existing_dataset_and_macro_summaries(self):
        source = json.loads((ROOT / findings.SOURCES[0]).read_text())
        selected = self.summary["routes"]["harp"]["selected"]
        original = self.summary["routes"]["harp"]["original"]
        for row in source["summary"]:
            computed = selected["datasets"][row["dataset"]]["methods"][row["method"]]
            for metric in findings.METRICS:
                for statistic in ("mean", "sample_sd"):
                    self.assertAlmostEqual(computed["statistics"][metric][statistic], row["statistics"][metric][statistic], places=10)
                for reference in ("R0", "R1"):
                    self.assertAlmostEqual(computed["versus"][reference][metric]["mean"], row["deltas"][reference][metric]["mean"], places=10)
        for method in findings.HARP_METHODS:
            self.assertAlmostEqual(selected["macro"]["methods"][method]["hm"]["mean"], source["ten_main"][method]["macro"]["hm"], places=10)
            self.assertEqual(selected["macro"]["methods"][method]["dual_positive_dataset_count"], source["ten_main"][method]["dual_positive"])
        for method in ("R0", "R1", "HARP"):
            self.assertAlmostEqual(original["macro"]["methods"][method]["hm"]["mean"], source["original_main_macro"][method]["hm"], places=10)
        self.assertEqual(selected["macro"]["main_dual_positive_dataset_count"], 8)
        self.assertEqual(original["macro"]["main_dual_positive_dataset_count"], 8)
        self.assertLess(original["macro"]["comparisons"]["main_minus_r1"]["hm"]["mean"], 0)
        self.assertGreater(selected["macro"]["comparisons"]["main_minus_r1"]["hm"]["mean"], 0)

    def test_real_larp_matches_existing_macro_and_dataset_summaries(self):
        source = json.loads((ROOT / "results/larp/analysis.json").read_text())
        panel = self.summary["routes"]["larp"]["recorded"]
        for method in findings.LARP_METHODS:
            if method == "R0":
                continue
            for metric in findings.METRICS:
                actual = panel["macro"]["methods"][method][metric]
                expected = source["macro"][method]
                self.assertAlmostEqual(actual["mean"], expected["mean"][metric], places=10)
                self.assertAlmostEqual(actual["sample_sd"], expected["macro_sample_sd"][metric], places=10)
                for dataset in findings.DATASETS:
                    row = panel["datasets"][dataset]["methods"][method]
                    reference = source["stats"][dataset][method]
                    self.assertAlmostEqual(row["statistics"][metric]["mean"], reference["mean"][metric], places=10)
                    self.assertAlmostEqual(row["statistics"][metric]["sample_sd"], reference["sd"][metric], places=10)
                    self.assertAlmostEqual(row["versus"]["R1"][metric]["mean"], reference["delta_r1"][metric], places=10)
                    self.assertAlmostEqual(row["versus"]["R0"][metric]["mean"], reference["delta_r0"][metric], places=10)
            for dataset in findings.DATASETS:
                self.assertEqual(panel["datasets"][dataset]["methods"][method]["versus"]["R1"]["positive_hm_seed_count"], source["stats"][dataset][method]["positive_seeds_r1"]["hm"])
            self.assertEqual(panel["macro"]["methods"][method]["dual_positive_dataset_count"], len(source["macro"][method]["dual_positive_datasets"]))
        self.assertEqual(panel["macro"]["main_dual_positive_dataset_count"], 7)
        self.assertAlmostEqual(panel["macro"]["comparisons"]["main_minus_r1"]["hm"]["mean"], -0.3046315047896845, places=12)
        food_delta = panel["datasets"]["food101"]["comparisons"]["main_minus_r1"]["hm"]["mean"]
        self.assertGreater(food_delta, 0)
        self.assertLess(food_delta, 0.003)

    def test_route_specific_r1_not_shared(self):
        harp = self.summary["routes"]["harp"]["selected"]
        larp = self.summary["routes"]["larp"]["recorded"]
        self.assertNotAlmostEqual(harp["macro"]["methods"]["R1"]["hm"]["mean"], larp["macro"]["methods"]["R1"]["hm"]["mean"], places=6)
        rows = json.loads((ROOT / findings.SOURCES[1]).read_text())
        for dataset in findings.DATASETS:
            for seed_index, seed in enumerate(findings.SEEDS):
                roles = {row["variant"]: row for row in rows if row["dataset"] == dataset and row["seed"] == seed}
                self.assertAlmostEqual(larp["datasets"][dataset]["comparisons"]["main_minus_r1"]["hm"]["values"][seed_index], roles["O4-r2"]["hm"] - roles["R1"]["hm"], places=12)

    def test_original_records_required_not_inferred(self):
        source = json.loads((ROOT / findings.SOURCES[0]).read_text())
        indexed = findings.index_records(source["records"], findings.HARP_METHODS)
        source["original_eurosat_seed2"].pop()
        with self.assertRaisesRegex(ValueError, "Missing or duplicate original"):
            findings.original_harp_index(source, indexed)

    def test_source_hashes_and_default_read_only(self):
        expected = {source["path"]: source["sha256"] for source in self.summary["source_files"]}
        with mock.patch.object(findings, "write_findings") as writer, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(findings.main([]), 0)
        writer.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["mode"], "read_only")
        for rel, sha256 in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), sha256)

    def test_new_directory_only_deterministic_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "findings"
            names = findings.write_findings(self.summary, output)
            self.assertEqual(set(names), {"summary.json", "main_comparison.md", "main_hm_deltas.svg"})
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            self.assertEqual(json.loads(before["summary.json"]), self.summary)
            self.assertEqual(before["main_comparison.md"].decode(), findings.markdown(self.summary))
            self.assertEqual(before["main_hm_deltas.svg"].decode(), findings.svg(self.summary))
            with self.assertRaises(FileExistsError):
                findings.write_findings(self.summary, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_svg_all_datasets_original_and_near_zero_label(self):
        payload = findings.svg(self.summary)
        root = ET.fromstring(payload)
        text = " ".join(root.itertext())
        text_elements = [node.text for node in root.findall("{http://www.w3.org/2000/svg}text")]
        for label in findings.LABELS.values():
            self.assertEqual(text_elements.count(label), 2)
        self.assertIn("Original − R1", text)
        self.assertIn("Original − R0", text)
        self.assertEqual(text.count("hollow = HARP original"), 1)
        self.assertIn("Only filled markers: main/control. Values in pp.", text)
        self.assertIn("+0.002406", text)
        circles = root.findall("{http://www.w3.org/2000/svg}circle")
        self.assertEqual(len(circles), 88)  # 50 HARP + 30 LARP values + 8 legend keys.
        for circle in circles:
            self.assertGreater(float(circle.attrib["cx"]), 0)
            self.assertLess(float(circle.attrib["cx"]), 1700)

    def test_checked_in_generated_outputs_match_when_present(self):
        output = ROOT / "results/findings"
        if not output.exists():
            self.skipTest("Generated outputs have not been written yet")
        self.assertEqual(json.loads((output / "summary.json").read_text()), self.summary)
        self.assertEqual((output / "main_comparison.md").read_text(), findings.markdown(self.summary))
        self.assertEqual((output / "main_hm_deltas.svg").read_text(), findings.svg(self.summary))


if __name__ == "__main__":
    unittest.main()
