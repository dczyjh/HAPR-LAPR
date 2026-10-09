import json
from pathlib import Path
import tempfile
import unittest
from tools.report_results import read_runs, compare


def record(method="none"):
    return {"schema_version": 2, "status": "complete", "method": method,
            "dataset": "Caltech101", "seed": 1, "rep_weight": 0.,
            "epochs": 20, "comparison_signature": "same",
            "base": 80., "novel": 80., "hm": 80.}


class ResultIntegrityTest(unittest.TestCase):
    def test_duplicate_seed_does_not_count_as_multiple_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("kd_200", "kd_1000"):
                (root / name).mkdir()
                (root / name / "metrics.json").write_text(json.dumps(record()))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_runs(root, "harp", 0.)

    def test_different_teacher_or_protocol_cannot_be_paired(self):
        base, new = record(), record("harp")
        new["comparison_signature"] = "different"
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            compare({("none", "caltech101", 1): base, ("harp", "caltech101", 1): new}, "harp", [1])

    def test_smoke_results_are_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = record()
            row["epochs"] = 1
            (root / "metrics.json").write_text(json.dumps(row))
            self.assertEqual(read_runs(root, "harp", 0.), {})

    def test_nan_and_inconsistent_hm_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for hm in (float("nan"), 81.):
                row = record()
                row["hm"] = hm
                (root / "metrics.json").write_text(json.dumps(row))
                with self.assertRaises(ValueError):
                    read_runs(root, "harp", 0.)
