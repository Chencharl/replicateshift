import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from replicateshift.benchmark import feature_map, role_seeds, run


class BenchmarkTests(unittest.TestCase):
    def test_role_seeds_separate_data_and_fitting(self):
        roles = role_seeds(7)
        self.assertEqual(roles, role_seeds(7))
        self.assertEqual(len(roles), len(set(roles.values())))
        a = np.random.default_rng(roles["simulation"]).normal(size=12)
        b = np.random.default_rng(roles["logistic"]).normal(size=12)
        self.assertFalse(np.array_equal(a, b))

    def test_interaction_features_include_every_pair_once(self):
        x = np.array([[2., 3., 5.]])
        np.testing.assert_array_equal(feature_map(x, "interaction_logistic"), [[2, 3, 5, 6, 10, 15]])

    def test_cli_artifacts_metrics_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            args = argparse.Namespace(specimens=60, replicates=3, epochs=3, bootstrap=10,
                seeds=[8], shifts=[0., 1.5], alpha=.2, output=str(output))
            run(args)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            for name, expected in manifest["outputs_sha256"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)
            with (output / "predictions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3 * 2 * 12)
            with (output / "per_seed_metrics.csv").open() as handle:
                metric_rows = list(csv.DictReader(handle))
            for metric in metric_rows:
                subset = [r for r in rows if (r["model"], r["shift"]) == (metric["model"], metric["shift"])]
                coverage = np.mean([int(r["include_" + r["label"]]) for r in subset])
                self.assertAlmostEqual(coverage, float(metric["coverage"]))
            report = (output / "REPORT.md").read_text()
            self.assertIn("80% guarantee", report)
            self.assertIn("uses 1 seeds", report)
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == "__main__":
    unittest.main()
