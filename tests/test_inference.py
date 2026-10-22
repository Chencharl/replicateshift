import unittest
import numpy as np
from replicateshift.inference import (conformal_threshold, prediction_sets,
                                      metrics, paired_brier_bootstrap, wilson_interval)


class InferenceTests(unittest.TestCase):
    def test_exact_order_statistic(self):
        p = np.arange(1, 11) / 11
        self.assertAlmostEqual(conformal_threshold(p, np.zeros(10), .2), 9 / 11)

    def test_small_calibration_includes_all_labels(self):
        q = conformal_threshold(np.array([.1, .9]), np.array([0, 1]), .1)
        self.assertTrue(np.isinf(q))
        self.assertTrue(prediction_sets(np.array([0., .5, 1.]), q).all())

    def test_ties_are_included_and_empty_sets_reported(self):
        np.testing.assert_array_equal(prediction_sets(np.array([.5]), .5), [[True, True]])
        self.assertEqual(metrics(np.array([.5]), np.array([1]), .4)["empty_set_rate"], 1)

    def test_label_symmetry(self):
        p, y = np.array([.1, .3, .8, .7]), np.array([0, 1, 1, 0])
        self.assertAlmostEqual(conformal_threshold(p, y, .3), conformal_threshold(1-p, 1-y, .3))

    def test_paired_identical_models(self):
        result = paired_brier_bootstrap(np.array([.1, .8]), np.array([.1, .8]), np.array([0, 1]))
        self.assertEqual(result["ci_low"], 0)
        self.assertEqual(result["ci_high"], 0)

    def test_invalid_inputs(self):
        for p in (np.array([]), np.array([np.nan]), np.array([1.1])):
            with self.assertRaises(ValueError):
                conformal_threshold(p, np.zeros(len(p)))
        with self.assertRaises(ValueError):
            conformal_threshold(np.array([.1]), np.array([2]))

    def test_iid_marginal_coverage_over_simulated_calibration_sets(self):
        # Distribution-free rank check, not evidence for domain-shift coverage.
        rng = np.random.default_rng(81)
        hits = []
        for _ in range(500):
            p = rng.uniform(size=100)
            y = rng.binomial(1, p)
            q = conformal_threshold(p[:49], y[:49], .1)
            hits.extend(prediction_sets(p[49:], q)[np.arange(51), y[49:]])
        self.assertTrue(.88 < np.mean(hits) < .93)

    def test_wilson_edges(self):
        self.assertAlmostEqual(wilson_interval(0, 10)[0], 0)
        self.assertAlmostEqual(wilson_interval(10, 10)[1], 1)
        for successes, n, z in ((.5, 10, 1.96), (5, 10, -1), (5, 10, float("nan")), (True, 10, 1.96)):
            with self.assertRaises(ValueError):
                wilson_interval(successes, n, z)


if __name__ == "__main__":
    unittest.main()
