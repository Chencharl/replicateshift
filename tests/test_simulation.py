"""Statistical-unit and known-truth simulator contracts."""

import unittest

import numpy as np

from replicateshift.simulation import (
    Dataset,
    aggregate_specimens,
    generate_dataset,
    split_by_row,
    split_by_specimen,
)


class SimulationTests(unittest.TestCase):
    def test_reproducibility_and_specimen_labels(self):
        a = generate_dataset(n_specimens=30, seed=11)
        b = generate_dataset(n_specimens=30, seed=11)
        for name in ("X", "y", "specimen_id", "replicate_id"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        self.assertEqual(a.X.shape, (90, 12))
        self.assertTrue(np.all(a.y.reshape(30, 3) == a.y.reshape(30, 3)[:, :1]))
        np.testing.assert_array_equal(a.replicate_id[:6], [0, 1, 2, 0, 1, 2])

    def test_specimen_split_is_complete_disjoint_and_reproducible(self):
        data = generate_dataset(n_specimens=40)
        parts = split_by_specimen(data, seed=13)
        again = split_by_specimen(data, seed=13)
        np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(120))
        self.assertEqual([len(x) for x in parts], [72, 24, 24])
        groups = [set(data.specimen_id[rows]) for rows in parts]
        for i in range(3):
            np.testing.assert_array_equal(parts[i], again[i])
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])

    def test_row_split_negative_control_leaks_specimens(self):
        data = generate_dataset(n_specimens=100, n_replicates=4)
        parts = split_by_row(data, seed=3)
        np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(400))
        groups = [set(data.specimen_id[rows]) for rows in parts]
        self.assertGreater(len(groups[0] & groups[1]), 20)
        self.assertGreater(len(groups[0] & groups[2]), 20)

    def test_shift_preserves_latent_labels_and_nuisance(self):
        common = dict(n_specimens=20, seed=29)
        original = generate_dataset(**common)
        shifted = generate_dataset(**common, shift=2.0)
        np.testing.assert_array_equal(original.y, shifted.y)
        np.testing.assert_array_equal(original.specimen_id, shifted.specimen_id)
        np.testing.assert_array_equal(original.X[:, 3:], shifted.X[:, 3:])
        np.testing.assert_allclose(shifted.X[:, 0] - original.X[:, 0], 2.0)
        np.testing.assert_allclose(shifted.X[:, 2] - original.X[:, 2], 1.0)
        self.assertFalse(np.allclose(original.X[:, 1], shifted.X[:, 1]))

    def test_zero_noise_replicates_are_identical(self):
        data = generate_dataset(n_specimens=10, noise=0)
        for specimen in range(10):
            rows = data.X[data.specimen_id == specimen]
            np.testing.assert_array_equal(rows, np.repeat(rows[:1], 3, axis=0))

    def test_aggregation_handles_arbitrary_subset_without_hidden_rows(self):
        data = Dataset(
            X=np.array([[1., 2.], [3., 4.], [10., 20.], [30., 40.]]),
            y=np.array([0, 0, 1, 1]),
            specimen_id=np.array([8, 8, 2, 2]),
            replicate_id=np.array([0, 1, 0, 1]),
        )
        X, y, ids = aggregate_specimens(data, np.array([1, 2, 0]))
        np.testing.assert_array_equal(ids, [2, 8])
        np.testing.assert_array_equal(y, [1, 0])
        np.testing.assert_array_equal(X, [[10., 20.], [2., 3.]])

    def test_invalid_generator_inputs(self):
        cases = [dict(n_specimens=0), dict(n_specimens=2.5), dict(n_replicates=0),
                 dict(n_features=2), dict(noise=-1), dict(noise=np.nan),
                 dict(noise="bad"), dict(shift=np.inf), dict(shift=1j),
                 dict(seed=-1), dict(seed=True)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_dataset(**kwargs)

    def test_invalid_splits_and_indices(self):
        data = generate_dataset(n_specimens=10)
        for fractions in ((0.6, 0.2, 0.1), (1, 0, 0), (0.5, 0.5), (np.nan, 0.2, 0.2)):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                split_by_specimen(data, fractions=fractions)
        with self.assertRaises(ValueError):
            split_by_specimen(generate_dataset(n_specimens=2))
        for rows in (np.array([], dtype=int), np.array([0, 0]), np.array([-1]),
                     np.array([30]), np.array([0.5]), np.array([[0]])):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                aggregate_specimens(data, rows)

    def test_dataset_rejects_inconsistent_labels_or_duplicate_replicates(self):
        common = dict(X=np.zeros((2, 3)), specimen_id=np.array([1, 1]))
        with self.assertRaises(ValueError):
            Dataset(**common, y=np.array([0, 1]), replicate_id=np.array([0, 1]))
        with self.assertRaises(ValueError):
            Dataset(**common, y=np.array([0, 0]), replicate_id=np.array([0, 0]))


if __name__ == "__main__":
    unittest.main()
