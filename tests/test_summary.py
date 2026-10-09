"""Synthetic JSON fixtures only: no research scores, images or GPU evaluation."""
import copy
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("summary_under_test", ROOT / "tools/summarize_run.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def fixture(root, route, seed, variant, base=80., novel=60., status="passed", **updates):
    """Clearly synthetic exact integer-count scores, not new experimental data."""
    check = {"row_scaled_failed_elements": 0, "hard_metric": "symmetric_per_example_infinity_norm"}
    value = dict(route=route, dataset="dtd", seed=seed, variant=variant, status=status,
                 base=base, novel=novel, hm=2 * base * novel / (base + novel) if base + novel else 0.,
                 selected_epoch=7, epochs=20, kd=200, canonical_evaluation="first_independent_reload",
                 counts_base_novel=[[100, int(base)], [100, int(novel)]],
                 checkpoint_sha256="a" * 64, best_last_sha256=["a" * 64, "b" * 64],
                 release_manifest_sha256="c" * 64,
                 reload_audit={name: copy.deepcopy(check) for name in ("native_val", "native_test", "selection_val")})
    value.update(updates)
    path = root / route / "dtd" / f"seed_{seed}" / "formal" / variant / "audited_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


class SummaryTests(unittest.TestCase):
    def test_three_seed_mean_sample_sd_and_paired_differences(self):
        with tempfile.TemporaryDirectory(prefix="summary-tiny-") as directory:
            root = Path(directory)
            roots = [root / name for name in ("main", "baseline")]
            for seed in (1, 2, 3):
                fixture(roots[0], "harp", seed, "r1", 80 + seed, 60 + seed)
                fixture(roots[0], "harp", seed, "harp", 81 + seed, 62 + seed)
                fixture(roots[1], "r0", seed, "r0", 79 + seed, 59 + seed)
            result = summary.summarize(roots)
            group = next(g for g in result["groups"] if g["variant"] == "harp")
            self.assertEqual(group["status"], "complete_three_seed")
            self.assertEqual(group["three_seed"]["base"], {"mean": 83., "sample_sd": 1.})
            self.assertEqual(group["comparisons"]["r1"]["three_seed_deltas_pp"]["base"], {"mean": 1., "sample_sd": 0.})
            self.assertEqual(group["comparisons"]["r0"]["three_seed_deltas_pp"]["novel"], {"mean": 3., "sample_sd": 0.})
            hm = [2 * (81 + s) * (62 + s) / (143 + 2 * s) for s in (1, 2, 3)]
            self.assertAlmostEqual(group["three_seed"]["hm"]["mean"], statistics.mean(hm))
            self.assertAlmostEqual(group["three_seed"]["hm"]["sample_sd"], statistics.stdev(hm))

    def test_failed_and_missing_seeds_never_make_three_seed_mean(self):
        with tempfile.TemporaryDirectory(prefix="summary-tiny-") as directory:
            root = Path(directory)
            fixture(root, "harp", 1, "harp")
            fixture(root, "harp", 2, "harp", base=99., novel=99., status="failed")
            fixture(root, "harp", 3, "harp", status="recorded_before_hard_checks")
            result = summary.summarize([root])
            self.assertEqual(result["counts"]["passed_formal"], 1)
            self.assertEqual(result["counts"]["excluded_nonpassed"], 2)
            self.assertEqual(result["groups"][0]["missing_seeds"], [2, 3])
            self.assertIsNone(result["groups"][0]["three_seed"])
            self.assertIsNone(result["groups"][0]["comparisons"]["r1"]["three_seed_deltas_pp"])
            self.assertIn("缺 passed seed 2,3", summary.markdown(result))
            self.assertNotIn("99.0000", summary.markdown(result))

    def test_route_specific_r1_and_missing_reference(self):
        with tempfile.TemporaryDirectory(prefix="summary-tiny-") as directory:
            root = Path(directory)
            for seed in (1, 2, 3):
                fixture(root, "larp", seed, "o4-r2")
                fixture(root, "harp", seed, "r1")
            fixture(root, "larp", 1, "r1")
            result = summary.summarize([root])
            group = next(g for g in result["groups"] if g["variant"] == "o4-r2")
            self.assertEqual(group["comparisons"]["r1"]["missing_reference_seeds"], [2, 3])
            self.assertIsNone(group["comparisons"]["r1"]["three_seed_deltas_pp"])
            self.assertEqual(group["comparisons"]["r0"]["missing_reference_seeds"], [1, 2, 3])

    def test_duplicate_keys_fail_even_when_one_failed(self):
        with tempfile.TemporaryDirectory(prefix="summary-tiny-") as directory:
            root = Path(directory)
            fixture(root / "a", "harp", 1, "harp", status="failed")
            fixture(root / "b", "harp", 1, "harp")
            with self.assertRaisesRegex(ValueError, "Duplicate result key"):
                summary.summarize([root / "a", root / "b"])

    def test_evaluation_copy_is_not_counted_twice(self):
        with tempfile.TemporaryDirectory(prefix="summary-tiny-") as directory:
            root = Path(directory)
            original = fixture(root, "harp", 1, "harp")
            copied = original.parents[2] / "evaluation" / "harp" / original.name
            copied.parent.mkdir(parents=True)
            copied.write_bytes(original.read_bytes())
            self.assertEqual(summary.summarize([root])["counts"]["passed_formal"], 1)

    def test_corrupt_advertised_passed_results_fail_closed(self):
        changes = [dict(hm=99.), dict(base=float("nan")), dict(epochs=19), dict(selected_epoch=0),
                   dict(canonical_evaluation="second_reload"), dict(kd=1000), dict(route="larp"),
                   dict(reload_audit={}), dict(counts_base_novel=[[100, 79], [100, 60]])]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory(prefix="summary-tiny-") as directory:
                path = fixture(Path(directory), "harp", 1, "harp")
                data = json.loads(path.read_text())
                data.update(change)
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    summary.summarize([directory])

    def test_release_drift_prevents_seed_pooling_or_reference_pairing(self):
        with tempfile.TemporaryDirectory(prefix="summary-tiny-") as directory:
            root = Path(directory)
            for seed in (1, 2, 3):
                fixture(root, "harp", seed, "harp", release_manifest_sha256=("d" if seed == 2 else "c") * 64)
                fixture(root, "harp", seed, "r1")
            result = summary.summarize([root])
            group = next(g for g in result["groups"] if g["variant"] == "harp")
            self.assertEqual(group["status"], "incompatible_release")
            self.assertIsNone(group["three_seed"])
            self.assertEqual(group["comparisons"]["r1"]["incompatible_release_seeds"], [2])

    def test_registration_includes_unstarted_roles(self):
        with tempfile.TemporaryDirectory(prefix="summary-tiny-") as directory:
            root = Path(directory)
            (root / "registration.json").write_text(json.dumps(dict(route="harp", plan=dict(
                route="harp", datasets=["dtd"], seeds=[1], variants=["r1", "harp"])) ))
            result = summary.summarize([root])
            self.assertEqual(len(result["groups"]), 2)
            self.assertTrue(all(group["missing_seeds"] == [1, 2, 3] for group in result["groups"]))
            self.assertTrue(all(group["registered_seeds"] == [1] for group in result["groups"]))

    def test_cli_markdown_and_json_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="summary-tiny-") as directory:
            root = Path(directory)
            fixture(root / "run", "harp", 1, "r1")
            destination = root / "report.json"
            command = [sys.executable, "-B", str(ROOT / "tools/summarize_run.py"), "--run-root", str(root / "run"),
                       "--output", str(destination)]
            result = subprocess.run(command, cwd="/", capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("# 新运行结果汇总", result.stdout)
            self.assertEqual(json.loads(destination.read_text())["counts"]["passed_formal"], 1)
            original = destination.read_bytes()
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(destination.read_bytes(), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
