"""Numerical and specimen-level behavioral checks for the Deep Sets model."""

import copy
import unittest

import numpy as np

from replicateshift.set_model import DeepSetClassifier, SetStandardizer


def small_data():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(9, 4, 3))
    mask = np.arange(4)[None, :] < np.array([1, 2, 3, 4, 2, 3, 4, 1, 2])[:, None]
    X[~mask] = np.nan
    y = np.array([0., 1., 0., 1., 1., 1., 0., 0., 1.])
    return X, mask, y


class SetStandardizerTests(unittest.TestCase):
    def test_statistics_weight_specimens_equally_and_ignore_padding(self):
        X = np.array([[[0., 7.], [np.nan, np.inf]], [[10., 7.], [14., 7.]]])
        mask = np.array([[True, False], [True, True]])
        scaler = SetStandardizer().fit(X, mask)
        np.testing.assert_allclose(scaler.mean_, [6., 7.])
        np.testing.assert_allclose(scaler.scale_, [np.sqrt(38.), 1.])
        standardized = scaler.transform(X, mask)
        np.testing.assert_array_equal(standardized[~mask], 0.)
        self.assertTrue(np.isfinite(standardized).all())
        # Duplicating one specimen's observed repeats must not change its weight.
        expanded = np.array([[[0., 7.], [0., 7.]], [[10., 7.], [14., 7.]]])
        other = SetStandardizer().fit(expanded, np.ones((2, 2), dtype=bool))
        np.testing.assert_allclose(scaler.mean_, other.mean_)
        np.testing.assert_allclose(scaler.scale_, other.scale_)

    def test_statistics_stay_fixed_for_new_specimens(self):
        X, mask, _ = small_data()
        scaler = SetStandardizer().fit(X, mask)
        mean, scale = scaler.mean_.copy(), scaler.scale_.copy()
        changed = X + 100.
        result = scaler.transform(changed, mask)
        np.testing.assert_array_equal(scaler.mean_, mean)
        np.testing.assert_array_equal(scaler.scale_, scale)
        self.assertGreater(result[mask].mean(), 40.)

    def test_state_roundtrip_and_copy_isolation(self):
        X, mask, _ = small_data()
        scaler = SetStandardizer().fit(X, mask)
        state = scaler.get_state()
        restored = SetStandardizer.from_state(state)
        np.testing.assert_array_equal(scaler.transform(X, mask), restored.transform(X, mask))
        state["mean"][:] = 999.
        state["scale"][:] = 999.
        np.testing.assert_array_equal(scaler.transform(X, mask), restored.transform(X, mask))
        with self.assertRaises(ValueError):
            SetStandardizer.from_state({"schema_version": 1, "mean": [0.], "scale": [0.]})
        with self.assertRaises(ValueError):
            SetStandardizer.from_state({"schema_version": 2, "mean": [0.], "scale": [1.]})
        with self.assertRaises(ValueError):
            SetStandardizer.from_state({"schema_version": True, "mean": [0.], "scale": [1.]})
        with self.assertRaises(ValueError):
            SetStandardizer.from_state({"schema_version": 1, "mean": [0.], "scale": [1.], "extra": 1})
        with self.assertRaises(RuntimeError):
            SetStandardizer().get_state()

    def test_invalid_input_and_unfitted_transform(self):
        with self.assertRaises(RuntimeError):
            SetStandardizer().transform([[[1.]]], [[True]])
        bad_inputs = [
            (np.ones((2, 2)), np.ones((2, 2), dtype=bool)),
            (np.ones((2, 0, 3)), np.ones((2, 0), dtype=bool)),
            (np.ones((2, 2, 3)), np.ones((2, 2), dtype=int)),
            (np.ones((2, 2, 3)), np.ones((2, 1), dtype=bool)),
            (np.ones((2, 2, 3)), np.array([[True, True], [False, False]])),
            (np.array([[[np.nan]]]), np.array([[True]])),
            (np.array([[[np.inf]]]), np.array([[True]])),
        ]
        for X, mask in bad_inputs:
            with self.subTest(shape=X.shape), self.assertRaises(ValueError):
                SetStandardizer().fit(X, mask)
        scaler = SetStandardizer().fit(np.ones((2, 2, 3)), np.ones((2, 2), dtype=bool))
        with self.assertRaises(ValueError):
            scaler.transform(np.ones((1, 2, 2)), np.ones((1, 2), dtype=bool))


class DeepSetTests(unittest.TestCase):
    def test_every_parameter_gradient_matches_central_finite_difference(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(4, 3, 2))
        mask = np.array([[True, False, False], [True, True, False],
                         [True, True, True], [False, True, False]])
        X[~mask] = np.nan
        y = np.array([0., 1., 1., 0.])
        model = DeepSetClassifier(seed=4, hidden=(3, 2), head_width=2, l2=.07)
        parameters = model._initialize_parameters(2)
        _, analytical = model.loss_and_gradients(X, mask, y, parameters)
        epsilon = 1e-6
        for name, parameter in parameters.items():
            numerical = np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus, _ = model.loss_and_gradients(X, mask, y, parameters)
                parameter[index] = original - epsilon
                minus, _ = model.loss_and_gradients(X, mask, y, parameters)
                parameter[index] = original
                numerical[index] = (plus - minus) / (2 * epsilon)
            with self.subTest(parameter=name):
                np.testing.assert_allclose(analytical[name], numerical, rtol=3e-5, atol=2e-8)

    def test_permutation_invariance_at_inference_and_training(self):
        X, mask, y = small_data()
        rng = np.random.default_rng(20)
        orders = np.array([rng.permutation(X.shape[1]) for _ in range(len(X))])
        rows = np.arange(len(X))[:, None]
        permuted_X, permuted_mask = X[rows, orders], mask[rows, orders]
        model = DeepSetClassifier(seed=5, epochs=15).fit(X, mask, y)
        np.testing.assert_allclose(model.predict_proba(X, mask),
                                   model.predict_proba(permuted_X, permuted_mask), atol=1e-14)
        other = DeepSetClassifier(seed=5, epochs=15).fit(permuted_X, permuted_mask, y)
        np.testing.assert_allclose(model.predict_proba(X, mask), other.predict_proba(X, mask), atol=1e-13)
        for name, value in model.parameters_.items():
            np.testing.assert_allclose(value, other.parameters_[name], atol=1e-13)

    def test_padding_is_ignored_in_prediction_loss_and_gradient(self):
        X, mask, y = small_data()
        model = DeepSetClassifier(epochs=8).fit(X, mask, y)
        changed = X.copy()
        changed[~mask] = np.inf
        np.testing.assert_array_equal(model.predict_proba(X, mask), model.predict_proba(changed, mask))
        loss_a, grad_a = model.loss_and_gradients(X, mask, y)
        loss_b, grad_b = model.loss_and_gradients(changed, mask, y)
        self.assertEqual(loss_a, loss_b)
        for name in grad_a:
            np.testing.assert_array_equal(grad_a[name], grad_b[name])
        more_padding = np.pad(X, ((0, 0), (0, 3), (0, 0)), constant_values=np.nan)
        bigger_mask = np.pad(mask, ((0, 0), (0, 3)), constant_values=False)
        np.testing.assert_allclose(model.predict_proba(X, mask),
                                   model.predict_proba(more_padding, bigger_mask), atol=1e-14)

    def test_mean_pooling_is_invariant_to_whole_set_duplication(self):
        X, mask, y = small_data()
        model = DeepSetClassifier(epochs=5).fit(X, mask, y)
        np.testing.assert_allclose(model.predict_proba(X, mask),
                                   model.predict_proba(np.repeat(X, 2, axis=1),
                                                       np.repeat(mask, 2, axis=1)), atol=1e-14)

    def test_fit_is_deterministic_and_refit_starts_fresh(self):
        X, mask, y = small_data()
        first = DeepSetClassifier(seed=3, epochs=20).fit(X, mask, y)
        second = DeepSetClassifier(seed=3, epochs=20).fit(X, mask, y)
        expected = first.predict_proba(X, mask)
        np.testing.assert_array_equal(expected, second.predict_proba(X, mask))
        first.fit(X, mask, y)
        np.testing.assert_array_equal(expected, first.predict_proba(X, mask))
        self.assertEqual(first.best_epoch_, 20)
        self.assertEqual(len(first.train_loss_history_), 20)
        self.assertEqual(first.validation_loss_history_, [])
        self.assertLess(first.final_loss_, first.train_loss_history_[0])
        self.assertTrue(np.isfinite(expected).all())
        self.assertTrue(((expected >= 0) & (expected <= 1)).all())

    def test_validation_selects_bce_checkpoint_without_affecting_training(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(30, 2, 2))
        mask = np.ones((30, 2), dtype=bool)
        y = (X.mean(axis=1)[:, 0] > 0).astype(float)
        options = dict(seed=2, epochs=35, hidden=(5, 4), head_width=3, learning_rate=.03, l2=.1)
        # Deliberately conflicting labels ensure the retained checkpoint is not
        # simply the final state; this is a checkpoint mechanism test only.
        model = DeepSetClassifier(**options).fit(X, mask, y, validation=(X, mask, 1-y))
        final = DeepSetClassifier(**options).fit(X, mask, y)
        np.testing.assert_array_equal(model.train_loss_history_, final.train_loss_history_)
        self.assertEqual(model.best_epoch_, np.argmin(model.validation_loss_history_) + 1)
        self.assertLess(model.best_epoch_, options["epochs"])
        replay_options = {**options, "epochs": model.best_epoch_}
        replay = DeepSetClassifier(**replay_options).fit(X, mask, y)
        for name in model.parameters_:
            np.testing.assert_array_equal(model.parameters_[name], replay.parameters_[name])
        p = model.predict_proba(X, mask)
        bce = -np.mean((1-y)*np.log(p) + y*np.log1p(-p))
        self.assertAlmostEqual(bce, min(model.validation_loss_history_), places=13)
        penalized, _ = model.loss_and_gradients(X, mask, 1-y)
        self.assertGreater(penalized, bce)

    def test_learns_set_distribution_with_identical_means(self):
        rng = np.random.default_rng(28)
        labels = np.tile([0., 1.], 70)
        magnitude = np.where(labels == 1, 1.7, .2) + rng.uniform(-.05, .05, len(labels))
        # Every specimen has mean zero; only the repeat distribution separates
        # these toy classes. This tests learned encoding before aggregation.
        X = np.stack([-magnitude, magnitude], axis=1)[..., None]
        mask = np.ones(X.shape[:2], dtype=bool)
        train, test = np.arange(100), np.arange(100, 140)
        model = DeepSetClassifier(seed=4, hidden=(8, 6), head_width=6, epochs=150,
                                  learning_rate=.02, l2=.0001).fit(X[train], mask[train], labels[train])
        accuracy = np.mean((model.predict_proba(X[test], mask[test]) >= .5) == labels[test])
        self.assertGreater(accuracy, .95)

    def test_state_roundtrip_and_copy_isolation(self):
        X, mask, y = small_data()
        model = DeepSetClassifier(epochs=9).fit(X, mask, y, validation=(X, mask, y))
        state = model.get_state()
        restored = DeepSetClassifier.from_state(state)
        np.testing.assert_array_equal(model.predict_proba(X, mask), restored.predict_proba(X, mask))
        self.assertEqual(model.best_epoch_, restored.best_epoch_)
        self.assertEqual(model.train_loss_history_, restored.train_loss_history_)
        state["parameters"]["W1"][:] = 0.
        state["train_loss_history"][0] = 999.
        np.testing.assert_array_equal(model.predict_proba(X, mask), restored.predict_proba(X, mask))
        self.assertEqual(model.train_loss_history_, restored.train_loss_history_)

    def test_malformed_saved_states_are_rejected(self):
        X, mask, y = small_data()
        state = DeepSetClassifier(epochs=3).fit(X, mask, y).get_state()
        bad_states = []
        for key, value in (("schema_version", 99), ("schema_version", True),
                           ("extra", "unknown"), ("n_features", 0), ("best_epoch", 0),
                           ("best_epoch", 1), ("train_loss_history", [float("nan")]*3),
                           ("validation_loss_history", [1.])):
            bad_states.append({**state, key: value})
        for config_change in ({"extra": 1}, {"learning_rate": None}, {"l2": []}):
            bad_states.append({**state, "config": {**state["config"], **config_change}})
        bad_states.append({key: value for key, value in state.items() if key != "best_epoch"})
        bad_states.append({**state, "config": {}})
        bad_parameters = copy.deepcopy(state)
        bad_parameters["parameters"]["W1"][0, 0] = np.inf
        bad_states.append(bad_parameters)
        bad_shape = copy.deepcopy(state)
        bad_shape["parameters"]["W4"] = np.zeros((1, 1))
        bad_states.append(bad_shape)
        for bad in bad_states:
            with self.subTest(state=bad.keys()), self.assertRaises(ValueError):
                DeepSetClassifier.from_state(bad)

    def test_stable_bce_at_extreme_logits(self):
        model = DeepSetClassifier(hidden=(2, 2), head_width=2)
        parameters = model._initialize_parameters(1)
        parameters["W4"][:] = 0.
        parameters["b4"][:] = 1000.
        loss, gradients = model.loss_and_gradients(np.ones((2, 1, 1)), np.ones((2, 1), dtype=bool),
                                                  np.array([0., 1.]), parameters)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.isfinite(value).all() for value in gradients.values()))

    def test_rejects_invalid_configuration_labels_masks_and_validation(self):
        for options in ({"epochs": 0}, {"epochs": 1.5}, {"seed": -1}, {"seed": True},
                        {"learning_rate": 0}, {"l2": -1}, {"hidden": (2, 0)},
                        {"hidden": (2,)}, {"head_width": False}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                DeepSetClassifier(**options)
        X, mask, y = small_data()
        with self.assertRaises(RuntimeError):
            DeepSetClassifier().predict_proba(X, mask)
        with self.assertRaises(RuntimeError):
            DeepSetClassifier().get_state()
        for labels in (y[:-1], y[:, None], np.full(len(y), np.nan), np.full(len(y), 2.)):
            with self.assertRaises(ValueError):
                DeepSetClassifier(epochs=1).fit(X, mask, labels)
        with self.assertRaises(ValueError):
            DeepSetClassifier(epochs=1).fit(X, mask.astype(int), y)
        with self.assertRaises(ValueError):
            DeepSetClassifier(epochs=1).fit(X, np.zeros_like(mask), y)
        with self.assertRaises(ValueError):
            DeepSetClassifier(epochs=1).fit(X, mask, y, validation=(X[..., :1], mask, y))
        with self.assertRaises(ValueError):
            DeepSetClassifier(epochs=1).fit(X, mask, y, validation=(X, mask))
        fitted = DeepSetClassifier(epochs=1).fit(X, mask, y)
        with self.assertRaises(ValueError):
            fitted.predict_proba(X[..., :1], mask)


if __name__ == "__main__":
    unittest.main()
