"""Small, inspectable binary models implemented with NumPy.

The neural model is an ordinary multilayer perceptron with two tanh hidden
layers, a Bernoulli output, analytic backpropagation, and full-batch Adam.
This is an educational implementation of established algorithms, not a new
learning algorithm. Scores are not guaranteed to be calibrated probabilities.

Preprocessing is deliberately separate: fit a Standardizer on training rows,
then reuse it unchanged for calibration and test rows.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

Array = np.ndarray
Parameters = dict[str, Array]


def _features(X: Array) -> Array:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a nonempty two-dimensional feature matrix")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    return X


def _labels(y: Array, n_samples: int) -> Array:
    y = np.asarray(y, dtype=np.float64)
    if y.shape != (n_samples,):
        raise ValueError("y must be one-dimensional with one label per row")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y must contain only binary labels 0 and 1")
    return y


def _sigmoid(logits: Array) -> Array:
    # This form avoids exp overflow for large negative logits.
    result = np.empty_like(logits)
    nonnegative = logits >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exp_logits = np.exp(logits[~nonnegative])
    result[~nonnegative] = exp_logits / (1.0 + exp_logits)
    return result


class Standardizer:
    """Training-set mean and population-SD scaling; constant columns stay safe."""

    def fit(self, X: Array) -> Standardizer:
        X = _features(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0, ddof=0)
        self.scale_[self.scale_ == 0] = 1.0
        if not np.isfinite(self.mean_).all() or not np.isfinite(self.scale_).all():
            raise ValueError("Feature magnitudes exceed safe standardization range")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: Array) -> Array:
        if not hasattr(self, "mean_"):
            raise RuntimeError("Fit the standardizer before transforming data")
        X = _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than training data")
        transformed = (X - self.mean_) / self.scale_
        if not np.isfinite(transformed).all():
            raise ValueError("Standardization produced nonfinite values")
        return transformed


class _AdamBinaryClassifier:
    def __init__(
        self,
        seed: int = 0,
        epochs: int = 200,
        learning_rate: float = 0.03,
        l2: float = 0.001,
    ) -> None:
        if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(l2) or l2 < 0:
            raise ValueError("l2 must be finite and nonnegative")
        self.seed = seed
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)

    def _initialize_parameters(self, n_features: int) -> Parameters:
        raise NotImplementedError

    def _logits(self, X: Array, parameters: Mapping[str, Array]) -> Array:
        raise NotImplementedError

    def loss_and_gradients(
        self, X: Array, y: Array, parameters: Mapping[str, Array] | None = None
    ) -> tuple[float, Parameters]:
        raise NotImplementedError

    def _checked_parameters(
        self, X: Array, parameters: Mapping[str, Array] | None
    ) -> Mapping[str, Array]:
        if parameters is None:
            if not hasattr(self, "parameters_"):
                raise RuntimeError("Fit the classifier or supply explicit parameters")
            parameters = self.parameters_
        expected = self._parameter_shapes(X.shape[1])
        if set(parameters) != set(expected):
            raise ValueError(f"Expected parameter keys {sorted(expected)}")
        for name, shape in expected.items():
            value = parameters[name]
            if not isinstance(value, np.ndarray) or value.shape != shape:
                raise ValueError(f"Parameter {name} must be an array of shape {shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"Parameter {name} must contain only finite values")
        return parameters

    def _parameter_shapes(self, n_features: int) -> dict[str, tuple[int, ...]]:
        raise NotImplementedError

    def fit(self, X: Array, y: Array, validation: tuple[Array, Array] | None = None) -> _AdamBinaryClassifier:
        """Optimize the mean binary cross-entropy plus 0.5*l2*sum(weights**2).

        Each call starts afresh from the same seed; intercepts are unpenalized.
        Optional validation selects the minimum unpenalized validation-BCE
        checkpoint after an update. Validation never contributes gradients.
        Without validation, behavior matches the original fixed-epoch fit.
        """
        X = _features(X)
        y = _labels(y, len(X))
        if validation is not None:
            if not isinstance(validation, (tuple, list)) or len(validation) != 2:
                raise ValueError("Validation must contain features and labels")
            Xval = _features(validation[0])
            yval = _labels(validation[1], len(Xval))
            if Xval.shape[1] != X.shape[1]:
                raise ValueError("Validation features must match training features")
        self.n_features_in_ = X.shape[1]
        self.parameters_ = self._initialize_parameters(self.n_features_in_)
        first = {name: np.zeros_like(value) for name, value in self.parameters_.items()}
        second = {name: np.zeros_like(value) for name, value in self.parameters_.items()}
        self.loss_history_: list[float] = []
        self.validation_loss_history_: list[float] = []
        self.best_epoch_ = self.epochs
        best_validation, best_parameters = float("inf"), None
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for step in range(1, self.epochs + 1):
            loss, gradients = self.loss_and_gradients(X, y, self.parameters_)
            if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients.values()):
                raise FloatingPointError("Nonfinite training loss or gradient; inspect feature scaling")
            self.loss_history_.append(loss)
            for name, gradient in gradients.items():
                first[name] = beta1 * first[name] + (1.0 - beta1) * gradient
                second[name] = beta2 * second[name] + (1.0 - beta2) * gradient**2
                corrected_first = first[name] / (1.0 - beta1**step)
                corrected_second = second[name] / (1.0 - beta2**step)
                self.parameters_[name] -= (
                    self.learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
                )
            if validation is not None:
                logits = self._logits(Xval, self.parameters_)
                value = float(np.mean(np.logaddexp(0., logits) - yval * logits))
                if not np.isfinite(value):
                    raise FloatingPointError("Nonfinite validation loss")
                self.validation_loss_history_.append(value)
                if value < best_validation:
                    best_validation = value
                    self.best_epoch_ = step
                    best_parameters = {name: value.copy() for name, value in self.parameters_.items()}
        if best_parameters is not None:
            self.parameters_ = best_parameters
        self.final_loss_, _ = self.loss_and_gradients(X, y, self.parameters_)
        if not np.isfinite(self.final_loss_):
            raise FloatingPointError("Nonfinite final training loss")
        return self

    def predict_proba(self, X: Array) -> Array:
        """Return a one-dimensional array of class-1 scores in [0, 1]."""
        X = _features(X)
        parameters = self._checked_parameters(X, None)
        logits = self._logits(X, parameters)
        if not np.isfinite(logits).all():
            raise FloatingPointError("Nonfinite prediction logits; inspect feature scaling")
        return _sigmoid(logits)


class LogisticClassifier(_AdamBinaryClassifier):
    """Linear Bernoulli model, trained using the same optimizer as the MLP."""

    def _parameter_shapes(self, n_features: int) -> dict[str, tuple[int, ...]]:
        return {"W": (n_features,), "b": (1,)}

    def _initialize_parameters(self, n_features: int) -> Parameters:
        rng = np.random.default_rng(self.seed)
        return {"W": rng.normal(0, 0.01, size=n_features), "b": np.zeros(1)}

    def _logits(self, X: Array, parameters: Mapping[str, Array]) -> Array:
        return X @ parameters["W"] + parameters["b"][0]

    def loss_and_gradients(
        self, X: Array, y: Array, parameters: Mapping[str, Array] | None = None
    ) -> tuple[float, Parameters]:
        X = _features(X)
        y = _labels(y, len(X))
        p = self._checked_parameters(X, parameters)
        logits = self._logits(X, p)
        loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
        loss += 0.5 * self.l2 * np.sum(p["W"] ** 2)
        delta = (_sigmoid(logits) - y) / len(X)
        gradients = {"W": X.T @ delta + self.l2 * p["W"], "b": np.array([delta.sum()])}
        return float(loss), gradients


class MLPClassifier(_AdamBinaryClassifier):
    """Two-hidden-layer tanh neural network with explicit backpropagation."""

    def __init__(
        self,
        seed: int = 0,
        epochs: int = 160,
        learning_rate: float = 0.01,
        l2: float = 0.001,
        hidden: tuple[int, int] = (32, 16),
    ) -> None:
        super().__init__(seed, epochs, learning_rate, l2)
        if len(hidden) != 2 or any(
            isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width < 1
            for width in hidden
        ):
            raise ValueError("hidden must contain two positive integer widths")
        self.hidden = tuple(int(width) for width in hidden)

    def _parameter_shapes(self, n_features: int) -> dict[str, tuple[int, ...]]:
        h1, h2 = self.hidden
        return {"W1": (n_features, h1), "b1": (h1,), "W2": (h1, h2), "b2": (h2,),
                "W3": (h2, 1), "b3": (1,)}

    def _initialize_parameters(self, n_features: int) -> Parameters:
        rng = np.random.default_rng(self.seed)
        parameters = {}
        widths = (n_features, *self.hidden, 1)
        for layer, (n_in, n_out) in enumerate(zip(widths[:-1], widths[1:]), start=1):
            bound = np.sqrt(6.0 / (n_in + n_out))
            parameters[f"W{layer}"] = rng.uniform(-bound, bound, size=(n_in, n_out))
            parameters[f"b{layer}"] = np.zeros(n_out)
        return parameters

    @staticmethod
    def _forward(X: Array, p: Mapping[str, Array]) -> tuple[Array, Array, Array]:
        h1 = np.tanh(X @ p["W1"] + p["b1"])
        h2 = np.tanh(h1 @ p["W2"] + p["b2"])
        logits = (h2 @ p["W3"] + p["b3"]).ravel()
        return h1, h2, logits

    def _logits(self, X: Array, parameters: Mapping[str, Array]) -> Array:
        return self._forward(X, parameters)[2]

    def loss_and_gradients(
        self, X: Array, y: Array, parameters: Mapping[str, Array] | None = None
    ) -> tuple[float, Parameters]:
        """Evaluate stable BCE and its gradient, optionally at supplied parameters.

        Parameters have keys W1, b1, W2, b2, W3, b3. This explicit interface
        enables central finite-difference checks independent of Adam training.
        """
        X = _features(X)
        y = _labels(y, len(X))
        p = self._checked_parameters(X, parameters)
        h1, h2, logits = self._forward(X, p)
        loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
        loss += 0.5 * self.l2 * sum(np.sum(p[f"W{i}"] ** 2) for i in (1, 2, 3))
        d3 = ((_sigmoid(logits) - y) / len(X))[:, None]
        d2 = (d3 @ p["W3"].T) * (1.0 - h2**2)
        d1 = (d2 @ p["W2"].T) * (1.0 - h1**2)
        gradients = {
            "W1": X.T @ d1 + self.l2 * p["W1"], "b1": d1.sum(axis=0),
            "W2": h1.T @ d2 + self.l2 * p["W2"], "b2": d2.sum(axis=0),
            "W3": h2.T @ d3 + self.l2 * p["W3"], "b3": d3.sum(axis=0),
        }
        return float(loss), gradients


class NearestNeighborClassifier:
    """One-nearest-neighbor control exposing memorization of repeated units.

    Scores are hard 0/1 labels and should not be interpreted as calibrated risk.
    Exact distance ties resolve to the earliest training row, deterministically.
    """

    def fit(self, X: Array, y: Array) -> NearestNeighborClassifier:
        X = _features(X)
        y = _labels(y, len(X))
        self.X_ = X.copy()
        self.y_ = y.copy()
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X: Array) -> Array:
        if not hasattr(self, "X_"):
            raise RuntimeError("Fit the classifier before predicting")
        X = _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than training data")
        predictions = np.empty(len(X))
        # Bound the temporary distance tensor; the intended demo has small data.
        for start in range(0, len(X), 128):
            query = X[start:start + 128]
            differences = query[:, None, :] - self.X_[None, :, :]
            distances = np.sum(differences**2, axis=2)
            if not np.isfinite(distances).all():
                raise FloatingPointError("Nonfinite distances; inspect feature scaling")
            predictions[start:start + len(query)] = self.y_[np.argmin(distances, axis=1)]
        return predictions
