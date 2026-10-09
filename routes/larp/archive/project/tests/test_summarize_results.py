import tempfile
import unittest
from pathlib import Path

from tools.compare_methods import bootstrap_interval
from tools.summarize_results import infer_run, parse_log


class SummarizeResultsTest(unittest.TestCase):
    def test_parses_best_base_and_final_novel(self):
        content = """\
Evaluate on the *val* set
* accuracy: 70.00%
Evaluate on the *val* set
* accuracy: 72.50%
Evaluate on the *test* set
* accuracy: 65.00%
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.txt"
            path.write_text(content)
            base, novel, harmonic_mean = parse_log(path)
        self.assertEqual(base, 72.5)
        self.assertEqual(novel, 65.0)
        self.assertAlmostEqual(harmonic_mean, 2 * 72.5 * 65 / (72.5 + 65))

    def test_infers_current_output_layout(self):
        root = Path("output/base2new")
        path = root / "larp/caltech101/kd_1000/rep_0.05/seed_1/log.txt"
        self.assertEqual(
            infer_run(path, root),
            ("larp", "caltech101", "1000", "0.05", 1),
        )

    def test_bootstrap_interval_is_exact_for_constant_values(self):
        self.assertEqual(bootstrap_interval([0.25] * 10), (0.25, 0.25))


if __name__ == "__main__":
    unittest.main()
