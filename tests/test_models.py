"""Numerical and behavioral tests for inspectable NumPy models."""

import unittest

import numpy as np

from replicateshift.models import (
    LogisticClassifier,
    MLPClassifier,
    NearestNeighborClassifier,
    Standardizer,
)


class StandardizerTests(unittest.TestCase):
    def test_training_statistics_remain_fixed_on_shifted_test_data(self):
        train = np.array([[1., 7.], [3., 7.], [5., 7.]])
        test = np.array([[101., 9.], [103., 9.]])
        scaler = Standardizer().fit(train)
        original_mean, original_scale = scaler.mean_.copy(), scaler.scale_.copy()
        transformed = scaler.transform(test)
        np.testing.assert_array_equal(scaler.mean_, original_mean)
        np.testing.assert_array_equal(scaler.scale_, original_scale)
        np.testing.assert_allclose(scaler.transform(train).mean(axis=0), [0., 0.])
        np.testing.assert_allclose(transformed[:, 1], [2., 2.])
        self.assertGreater(transformed[0, 0], 50.)

    def test_invalid_input_and_unfitted_transform(self):
        with self.assertRaises(RuntimeError):
            Standardizer().transform([[1.]])
        for X in ([], [1., 2.], [[np.nan]], np.empty((2, 0))):
            with self.subTest(X=X), self.assertRaises(ValueError):
                Standardizer().fit(X)
        with self.assertRaises(ValueError):
            Standardizer().fit([[1., 2.]]).transform([[1.]])


class ModelTests(unittest.TestCase):
    @staticmethod
    def nonlinear_data():
        rng = np.random.default_rng(17)
        # Symmetric noisy XOR: a linear model cannot separate all four groups.
        centers = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        X = np.repeat(centers, 40, axis=0) + rng.normal(0, .13, size=(160, 2))
        y = np.repeat([0., 1., 1., 0.], 40)
        return X, y

    def test_mlp_gradients_match_central_finite_difference(self):
        rng = np.random.default_rng(9)
        X, y = rng.normal(size=(7, 3)), np.array([0., 1., 1., 0., 1., 0., 1.])
        model = MLPClassifier(seed=2, hidden=(4, 3), l2=.07)
        parameters = model._initialize_parameters(3)
        _, analytical = model.loss_and_gradients(X, y, parameters)
        epsilon = 1e-6
        for name, parameter in parameters.items():
            numerical = np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus, _ = model.loss_and_gradients(X, y, parameters)
                parameter[index] = original - epsilon
                minus, _ = model.loss_and_gradients(X, y, parameters)
                parameter[index] = original
                numerical[index] = (plus - minus) / (2 * epsilon)
            with self.subTest(parameter=name):
                np.testing.assert_allclose(analytical[name], numerical, rtol=2e-5, atol=2e-8)

    def test_logistic_gradients_and_extreme_logits(self):
        X = np.array([[-1., .4], [.5, -1.], [1., .7]])
        y = np.array([0., 1., 1.])
        model = LogisticClassifier(l2=.03)
        parameters = {"W": np.array([.3, -.2]), "b": np.array([.1])}
        _, analytical = model.loss_and_gradients(X, y, parameters)
        for name, parameter in parameters.items():
            numerical = np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + 1e-6
                plus, _ = model.loss_and_gradients(X, y, parameters)
                parameter[index] = original - 1e-6
                minus, _ = model.loss_and_gradients(X, y, parameters)
                parameter[index] = original
                numerical[index] = (plus - minus) / 2e-6
            np.testing.assert_allclose(analytical[name], numerical, rtol=1e-5, atol=1e-8)
        parameters = {"W": np.array([1000.]), "b": np.array([0.])}
        loss, gradients = model.loss_and_gradients(np.array([[-1.], [1.]]), [1., 0.], parameters)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.isfinite(g).all() for g in gradients.values()))

    def test_training_is_deterministic_and_produces_finite_probabilities(self):
        X, y = self.nonlinear_data()
        for factory in (LogisticClassifier, MLPClassifier):
            with self.subTest(model=factory.__name__):
                first = factory(seed=3, epochs=40).fit(X, y)
                second = factory(seed=3, epochs=40).fit(X, y)
                predictions = first.predict_proba(X)
                np.testing.assert_array_equal(predictions, second.predict_proba(X))
                self.assertEqual(predictions.shape, (len(X),))
                self.assertTrue(np.isfinite(predictions).all())
                self.assertTrue(((predictions >= 0) & (predictions <= 1)).all())
                self.assertLess(first.final_loss_, first.loss_history_[0])
                first.fit(X, y)
                np.testing.assert_array_equal(predictions, first.predict_proba(X))

    def test_mlp_learns_nonlinear_rule_on_unseen_points(self):
        X, y = self.nonlinear_data()
        train, test = np.arange(0, len(X), 2), np.arange(1, len(X), 2)
        model = MLPClassifier(seed=5, hidden=(12, 8), epochs=160).fit(X[train], y[train])
        accuracy = np.mean((model.predict_proba(X[test]) >= .5) == y[test])
        linear = LogisticClassifier(seed=5).fit(X[train], y[train])
        linear_accuracy = np.mean((linear.predict_proba(X[test]) >= .5) == y[test])
        self.assertGreater(accuracy, .95)
        self.assertGreater(accuracy - linear_accuracy, .2)

    def test_invalid_inputs_fail_clearly(self):
        for factory in (LogisticClassifier, MLPClassifier):
            with self.subTest(model=factory.__name__):
                with self.assertRaises(RuntimeError):
                    factory().predict_proba([[1.]])
                for y in ([0., 2.], [[0.], [1.]], [np.nan, 0.], [0.]):
                    with self.assertRaises(ValueError):
                        factory().fit([[1.], [2.]], y)
                fitted = factory(epochs=2).fit([[1.], [2.]], [0., 1.])
                with self.assertRaises(ValueError):
                    fitted.predict_proba([[1., 2.]])
                with self.assertRaises(ValueError):
                    fitted.predict_proba([[np.inf]])
                for options in ({"epochs": 0}, {"epochs": 1.5}, {"learning_rate": 0}, {"l2": -1}):
                    with self.assertRaises(ValueError):
                        factory(**options)
        with self.assertRaises(ValueError):
            MLPClassifier(hidden=(3, 0))

    def test_nearest_neighbor_memorizes_exact_replicates(self):
        train = np.array([[0., 1.], [3., 2.], [-1., 0.]])
        labels = np.array([0., 1., 1.])
        model = NearestNeighborClassifier().fit(train, labels)
        np.testing.assert_array_equal(model.predict_proba(train), labels)
        train[:] = 100.
        labels[:] = 0.
        np.testing.assert_array_equal(model.predict_proba([[3., 2.], [-1., 0.]]), [1., 1.])
        with self.assertRaises(ValueError):
            model.predict_proba([[1.]])


if __name__ == "__main__":
    unittest.main()
