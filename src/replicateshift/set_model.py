"""Permutation-invariant binary prediction from sets of technical repeats.

The established Deep Sets construction is ``rho(mean(phi(repeat)))``: two
tanh encoder layers, a masked mean, and a tanh decoder with a Bernoulli
output. Mean pooling makes the score invariant to repeat order and to
duplicating an entire repeat set. It does not supply a calibrated risk or a
new statistical guarantee. Training minimizes specimen-weighted BCE plus
weight-only L2 regularization using full-batch Adam.

Arrays have shape (specimens, padded repeats, features). Boolean masks mark
observed repeats. Every specimen must have at least one observed repeat;
padding is ignored even if it contains NaN or infinity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .models import _labels, _sigmoid

Array = np.ndarray
Parameters = dict[str, Array]


def _sets(X: Array, mask: Array) -> tuple[Array, Array]:
    X = np.asarray(X, dtype=np.float64)
    mask = np.asarray(mask)
    if X.ndim != 3 or any(size == 0 for size in X.shape):
        raise ValueError("X must be a nonempty (specimens, repeats, features) array")
    if mask.dtype != np.bool_ or mask.shape != X.shape[:2]:
        raise ValueError("mask must be a boolean array with shape X.shape[:2]")
    if not mask.any(axis=1).all():
        raise ValueError("Every specimen must have at least one observed repeat")
    if not np.isfinite(X[mask]).all():
        raise ValueError("Observed repeats must contain only finite values")
    # Do not multiply padding by zero: IEEE arithmetic leaves NaN * 0 as NaN.
    return np.where(mask[..., None], X, 0.0), mask


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _check_state_keys(state: Any, expected: set[str], name: str) -> None:
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError(f"{name} must have exactly these keys: {sorted(expected)}")


def _check_schema(state: Mapping[str, Any], name: str) -> None:
    version = state["schema_version"]
    if isinstance(version, bool) or not isinstance(version, (int, np.integer)) or version != 1:
        raise ValueError(f"Unsupported {name} state schema")


class SetStandardizer:
    """Scale each feature using observed training repeats; padding stays zero.

    Every specimen has equal total weight in these preprocessing statistics;
    repeats share their specimen's weight. The variance is the specimen-mean
    of within-specimen mean squared deviations from the global training mean.
    The classifier's loss also gives each independent specimen equal weight.
    Fit this object on the training split only, then freeze it for all other
    splits. Constant training features use a scale of one.
    """

    def fit(self, X: Array, mask: Array) -> SetStandardizer:
        X, mask = _sets(X, mask)
        weights = mask / mask.sum(axis=1, keepdims=True)
        with np.errstate(over="ignore", invalid="ignore"):
            mean = np.sum(X * weights[..., None], axis=1).mean(axis=0)
            centered = np.where(mask[..., None], X - mean, 0.0)
            variance = np.sum(centered**2 * weights[..., None], axis=1).mean(axis=0)
            scale = np.sqrt(variance)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("Feature magnitudes exceed safe standardization range")
        scale[scale == 0.0] = 1.0
        self.mean_, self.scale_ = mean, scale
        self.n_features_in_ = X.shape[2]
        return self

    def transform(self, X: Array, mask: Array) -> Array:
        if not hasattr(self, "mean_"):
            raise RuntimeError("Fit the standardizer before transforming data")
        X, mask = _sets(X, mask)
        if X.shape[2] != self.n_features_in_:
            raise ValueError("X has a different number of features than training data")
        transformed = np.zeros_like(X)
        with np.errstate(over="ignore", invalid="ignore"):
            transformed[mask] = (X[mask] - self.mean_) / self.scale_
        if not np.isfinite(transformed).all():
            raise ValueError("Standardization produced nonfinite values")
        return transformed

    def get_state(self) -> dict[str, Any]:
        """Return independent arrays suitable for a non-pickle NPZ archive."""
        if not hasattr(self, "mean_"):
            raise RuntimeError("Fit the standardizer before saving its state")
        return {"schema_version": 1, "mean": self.mean_.copy(), "scale": self.scale_.copy()}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> SetStandardizer:
        _check_state_keys(state, {"schema_version", "mean", "scale"}, "Standardizer state")
        _check_schema(state, "standardizer")
        try:
            mean = np.asarray(state["mean"], dtype=np.float64)
            scale = np.asarray(state["scale"], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Standardizer mean and scale must be numeric arrays") from error
        if (mean.ndim != 1 or mean.size == 0 or scale.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                or not (scale > 0).all()):
            raise ValueError("Invalid standardizer mean or scale")
        model = cls()
        model.mean_, model.scale_ = mean.copy(), scale.copy()
        model.n_features_in_ = mean.size
        return model


class DeepSetClassifier:
    """Two-layer repeat encoder and nonlinear specimen-level decoder.

    Validation, when provided, selects the epoch with the lowest unpenalized
    validation BCE. Calibration and final evaluation labels must never be
    passed as validation data. There is no automatic early stopping or reuse
    of validation observations for parameter gradients.
    """

    def __init__(
        self,
        seed: int = 0,
        epochs: int = 120,
        learning_rate: float = 0.003,
        l2: float = 0.003,
        hidden: tuple[int, int] = (16, 12),
        head_width: int = 12,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.seed = int(seed)
        self.epochs = _positive_int(epochs, "epochs")
        if (isinstance(learning_rate, bool)
                or not isinstance(learning_rate, (int, float, np.integer, np.floating))
                or not np.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError("learning_rate must be finite and positive")
        if (isinstance(l2, bool) or not isinstance(l2, (int, float, np.integer, np.floating))
                or not np.isfinite(l2) or l2 < 0):
            raise ValueError("l2 must be finite and nonnegative")
        if not isinstance(hidden, (tuple, list)) or len(hidden) != 2:
            raise ValueError("hidden must contain two positive integer widths")
        self.hidden = tuple(_positive_int(width, "hidden width") for width in hidden)
        self.head_width = _positive_int(head_width, "head_width")
        self.learning_rate, self.l2 = float(learning_rate), float(l2)

    def _parameter_shapes(self, n_features: int) -> dict[str, tuple[int, ...]]:
        widths = (n_features, *self.hidden, self.head_width, 1)
        shapes = {}
        for layer, (n_in, n_out) in enumerate(zip(widths[:-1], widths[1:]), 1):
            shapes[f"W{layer}"] = (n_in, n_out)
            shapes[f"b{layer}"] = (n_out,)
        return shapes

    def _initialize_parameters(self, n_features: int) -> Parameters:
        rng = np.random.default_rng(self.seed)
        parameters = {}
        for name, shape in self._parameter_shapes(n_features).items():
            if name.startswith("W"):
                bound = np.sqrt(6.0 / sum(shape))
                parameters[name] = rng.uniform(-bound, bound, size=shape)
            else:
                parameters[name] = np.zeros(shape)
        return parameters

    def _checked_parameters(
        self, n_features: int, parameters: Mapping[str, Array] | None
    ) -> Mapping[str, Array]:
        if parameters is None:
            if not hasattr(self, "parameters_"):
                raise RuntimeError("Fit the classifier or supply explicit parameters")
            parameters = self.parameters_
        expected = self._parameter_shapes(n_features)
        if set(parameters) != set(expected):
            raise ValueError(f"Expected parameter keys {sorted(expected)}")
        for name, shape in expected.items():
            value = parameters[name]
            if not isinstance(value, np.ndarray) or value.shape != shape:
                raise ValueError(f"Parameter {name} must be an array of shape {shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"Parameter {name} must contain only finite values")
        return parameters

    @staticmethod
    def _forward(
        X: Array, mask: Array, p: Mapping[str, Array]
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        with np.errstate(over="ignore", invalid="ignore"):
            a1 = X @ p["W1"] + p["b1"]
            h1 = np.tanh(a1)
            a2 = h1 @ p["W2"] + p["b2"]
            h2 = np.tanh(a2)
            weights = mask / mask.sum(axis=1, keepdims=True)
            pooled = np.sum(h2 * weights[..., None], axis=1)
            a3 = pooled @ p["W3"] + p["b3"]
            h3 = np.tanh(a3)
            logits = (h3 @ p["W4"] + p["b4"]).ravel()
        if any(not np.isfinite(a).all() for a in (a1, a2, a3, logits)):
            raise FloatingPointError("Nonfinite activations; inspect feature scaling and parameters")
        return h1, h2, pooled, h3, logits, weights

    def _loss(self, logits: Array, y: Array, p: Mapping[str, Array]) -> float:
        loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
        loss += 0.5 * self.l2 * sum(np.sum(p[f"W{i}"] ** 2) for i in range(1, 5))
        return float(loss)

    def loss_and_gradients(
        self,
        X: Array,
        mask: Array,
        y: Array,
        parameters: Mapping[str, Array] | None = None,
    ) -> tuple[float, Parameters]:
        """Return specimen-mean BCE + 0.5*l2*sum(W**2), and all gradients."""
        X, mask = _sets(X, mask)
        y = _labels(y, len(X))
        p = self._checked_parameters(X.shape[2], parameters)
        h1, h2, pooled, h3, logits, weights = self._forward(X, mask, p)
        d4 = ((_sigmoid(logits) - y) / len(X))[:, None]
        d3 = (d4 @ p["W4"].T) * (1.0 - h3**2)
        dpool = d3 @ p["W3"].T
        d2 = dpool[:, None, :] * weights[..., None] * (1.0 - h2**2)
        d1 = (d2 @ p["W2"].T) * (1.0 - h1**2)
        gradients = {
            "W1": X.reshape(-1, X.shape[2]).T @ d1.reshape(-1, self.hidden[0]) + self.l2*p["W1"],
            "b1": d1.sum(axis=(0, 1)),
            "W2": h1.reshape(-1, self.hidden[0]).T @ d2.reshape(-1, self.hidden[1]) + self.l2*p["W2"],
            "b2": d2.sum(axis=(0, 1)),
            "W3": pooled.T @ d3 + self.l2*p["W3"],
            "b3": d3.sum(axis=0),
            "W4": h3.T @ d4 + self.l2*p["W4"],
            "b4": d4.sum(axis=0),
        }
        return self._loss(logits, y, p), gradients

    def fit(
        self,
        X: Array,
        mask: Array,
        y: Array,
        validation: tuple[Array, Array, Array] | None = None,
    ) -> DeepSetClassifier:
        """Fit afresh from the seed and optionally restore the best validation epoch.

        Histories describe the state after each Adam update. Epoch numbering is
        one-based. Strict improvement selects a checkpoint, so ties prefer the
        earlier epoch. Without validation the final epoch is retained.
        """
        X, mask = _sets(X, mask)
        y = _labels(y, len(X))
        if validation is not None:
            if not isinstance(validation, (tuple, list)) or len(validation) != 3:
                raise ValueError("validation must be (X, mask, y)")
            Xv, mv = _sets(validation[0], validation[1])
            yv = _labels(validation[2], len(Xv))
            if Xv.shape[2] != X.shape[2]:
                raise ValueError("Validation data must have the training feature count")
        self.n_features_in_ = X.shape[2]
        self.parameters_ = self._initialize_parameters(self.n_features_in_)
        first = {k: np.zeros_like(v) for k, v in self.parameters_.items()}
        second = {k: np.zeros_like(v) for k, v in self.parameters_.items()}
        self.train_loss_history_: list[float] = []
        self.validation_loss_history_: list[float] = []
        self.best_epoch_ = self.epochs
        best_loss, best_parameters = float("inf"), None
        for epoch in range(1, self.epochs + 1):
            loss, gradients = self.loss_and_gradients(X, mask, y, self.parameters_)
            if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients.values()):
                raise FloatingPointError("Nonfinite training loss or gradient; inspect feature scaling")
            for name, gradient in gradients.items():
                first[name] = 0.9*first[name] + 0.1*gradient
                second[name] = 0.999*second[name] + 0.001*gradient**2
                corrected_first = first[name] / (1.0 - 0.9**epoch)
                corrected_second = second[name] / (1.0 - 0.999**epoch)
                self.parameters_[name] -= self.learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            logits = self._forward(X, mask, self.parameters_)[4]
            train_loss = self._loss(logits, y, self.parameters_)
            if not np.isfinite(train_loss):
                raise FloatingPointError("Nonfinite training loss after update")
            self.train_loss_history_.append(train_loss)
            if validation is not None:
                logits_v = self._forward(Xv, mv, self.parameters_)[4]
                val_loss = float(np.mean(np.logaddexp(0.0, logits_v) - yv*logits_v))
                if not np.isfinite(val_loss):
                    raise FloatingPointError("Nonfinite validation loss after update")
                self.validation_loss_history_.append(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_parameters = {k: v.copy() for k, v in self.parameters_.items()}
                    self.best_epoch_ = epoch
        if best_parameters is not None:
            self.parameters_ = best_parameters
        self.best_validation_loss_ = None if validation is None else best_loss
        self.final_loss_ = self.train_loss_history_[self.best_epoch_ - 1]
        return self

    def predict_proba(self, X: Array, mask: Array) -> Array:
        """Return class-1 scores; these scores alone are not calibrated risks."""
        X, mask = _sets(X, mask)
        p = self._checked_parameters(X.shape[2], None)
        return _sigmoid(self._forward(X, mask, p)[4])

    def get_state(self) -> dict[str, Any]:
        """Return constructor metadata and copied arrays, without executable pickle."""
        if not hasattr(self, "parameters_"):
            raise RuntimeError("Fit the classifier before saving its state")
        return {
            "schema_version": 1,
            "config": {"seed": self.seed, "epochs": self.epochs,
                       "learning_rate": self.learning_rate, "l2": self.l2,
                       "hidden": list(self.hidden), "head_width": self.head_width},
            "n_features": self.n_features_in_,
            "parameters": {k: v.copy() for k, v in self.parameters_.items()},
            "best_epoch": self.best_epoch_,
            "train_loss_history": list(self.train_loss_history_),
            "validation_loss_history": list(self.validation_loss_history_),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> DeepSetClassifier:
        _check_state_keys(state, {"schema_version", "config", "n_features", "parameters",
                                 "best_epoch", "train_loss_history", "validation_loss_history"},
                          "Classifier state")
        _check_schema(state, "classifier")
        config = state["config"]
        _check_state_keys(config, {"seed", "epochs", "learning_rate", "l2", "hidden", "head_width"},
                          "Classifier config")
        model = cls(**config)
        model.n_features_in_ = _positive_int(state.get("n_features"), "n_features")
        parameters = state.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("Classifier state must contain parameters")
        try:
            converted = {k: np.asarray(v, dtype=np.float64) for k, v in parameters.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("Classifier parameters must be numeric arrays") from error
        checked = model._checked_parameters(model.n_features_in_, converted)
        model.parameters_ = {k: v.copy() for k, v in checked.items()}
        best_epoch = _positive_int(state.get("best_epoch"), "best_epoch")
        histories = []
        for key in ("train_loss_history", "validation_loss_history"):
            try:
                history = np.asarray(state[key], dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid {key}") from error
            if history.ndim != 1 or not np.isfinite(history).all() or (history < 0).any():
                raise ValueError(f"Invalid {key}")
            histories.append(history.tolist())
        train, validation = histories
        if len(train) != model.epochs or len(validation) not in (0, model.epochs):
            raise ValueError("History lengths must match configured epochs")
        if best_epoch > model.epochs or (not validation and best_epoch != model.epochs):
            raise ValueError("Invalid best_epoch for saved histories")
        if validation and best_epoch != int(np.argmin(validation)) + 1:
            raise ValueError("best_epoch does not match validation loss history")
        model.best_epoch_ = best_epoch
        model.train_loss_history_, model.validation_loss_history_ = train, validation
        model.final_loss_ = train[best_epoch - 1]
        model.best_validation_loss_ = validation[best_epoch - 1] if validation else None
        return model
