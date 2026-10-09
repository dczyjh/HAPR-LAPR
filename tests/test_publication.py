"""Small filesystem/provenance tests; no research assets or model inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("asset_checker", ROOT / "tools/check_assets.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class PublicationTests(unittest.TestCase):
    def test_asset_checker_rejects_missing_size_digest_and_traversal(self):
        with tempfile.TemporaryDirectory(prefix="promptkd-assets-test-") as folder:
            root = Path(folder)
            path = root / "fixture.txt"
            path.write_bytes(b"offline fixture")
            row = {"kind": "data_metadata", "relative_path": "fixture.txt", "bytes": path.stat().st_size,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(checker.check_entry(root, row)["status"], "passed")
            for changed, expected in [({"relative_path": "missing"}, "missing"),
                                      ({"bytes": 0}, "size_mismatch"),
                                      ({"sha256": "0" * 64}, "sha256_mismatch"),
                                      ({"relative_path": "../outside"}, "invalid_manifest_path")]:
                self.assertEqual(checker.check_entry(root, dict(row, **changed))["status"], expected)

    def test_existing_result_matrices_and_flags(self):
        harp = json.loads((ROOT / "results/harp/complete_results.json").read_text())
        larp = json.loads((ROOT / "results/larp/flat_results.json").read_text())
        self.assertEqual(len(harp["records"]), 180)
        self.assertEqual(len(larp), 150)
        self.assertEqual(len({(r["dataset"], r["seed"], r["method"]) for r in harp["records"]}), 180)
        self.assertEqual(len({(r["dataset"], r["seed"], r["variant"]) for r in larp}), 150)
        self.assertIn("original_eurosat_seed2", harp)
        self.assertIn("revision_selections", harp)
        self.assertTrue(harp["selection_after_results_seen"])
        self.assertFalse(harp["original_results_invalidated"])
        self.assertTrue(any(not r["same_machine_as_r1"] for r in larp))

    def test_result_copy_provenance_digests(self):
        rows = json.loads((ROOT / "metadata/publication_sources.json").read_text())["files"]
        for row in rows:
            self.assertEqual(hashlib.sha256((ROOT / row["target"]).read_bytes()).hexdigest(), row["published_sha256"])

    def test_asset_inventory(self):
        rows = json.loads((ROOT / "metadata/external_assets.json").read_text())["assets"]
        self.assertEqual(sum(x["kind"] == "pretrained" for x in rows), 12)
        self.assertEqual(sum(x["relative_path"].endswith(".json") for x in rows), 9)
        self.assertEqual(sum(x["relative_path"].endswith(".txt") for x in rows), 4)
        for row in rows:
            self.assertEqual(len(row["sha256"]), 64)

    def test_dispatcher_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory(prefix="promptkd-dispatch-test-") as folder:
            for route in ["harp", "larp"]:
                result = subprocess.run([sys.executable, "-B", str(ROOT / "project.py"), route, "--help"],
                                        cwd=folder, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([sys.executable, "-B", str(ROOT / "project.py"), "not-a-route"],
                                    cwd=folder, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
