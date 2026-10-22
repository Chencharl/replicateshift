"""Specimen-level split conformal prediction and conditional evaluation.

One score per independent specimen. These functions do not repair dependence
between specimens, distribution shift, or reuse of calibration labels for fitting.
"""
from __future__ import annotations

import math
import numpy as np


def _probabilities(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size == 0 or not np.isfinite(p).all():
        raise ValueError("Probabilities must be a nonempty finite vector.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must lie in [0, 1].")
    return p


def _labels(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y)
    if y.shape != (n,) or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be a matching binary vector.")
    return y.astype(int)


def conformal_threshold(p: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> float:
    """Finite-sample order statistic, including the +infinity sentinel.

    For k = ceil((n+1)*(1-alpha)), use the kth smallest score (one based).
    If k > n, all labels must be included. Never interpolate the quantile.
    """
    p = _probabilities(p)
    y = _labels(y, len(p))
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one.")
    scores = np.where(y == 1, 1 - p, p)
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    return float(np.partition(scores, k - 1)[k - 1]) if k <= len(scores) else math.inf


def prediction_sets(p: np.ndarray, threshold: float) -> np.ndarray:
    """Return n-by-2 inclusion flags for labels [0, 1]; ties are included."""
    p = _probabilities(p)
    if math.isnan(threshold) or threshold < 0:
        raise ValueError("threshold must be nonnegative or +infinity.")
    return np.column_stack((p, 1 - p)) <= threshold


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Approximate 95% binomial interval for independent test specimens.

    Conditional on the fitted predictor and calibration set; does not include
    variability from retraining or recalibration. Shared-batch dependence breaks
    the binomial interpretation.
    """
    if (isinstance(n, (bool, np.bool_)) or isinstance(successes, (bool, np.bool_))
            or not isinstance(n, (int, np.integer)) or not isinstance(successes, (int, np.integer))
            or n < 1 or not 0 <= successes <= n):
        raise ValueError("Require n > 0 and 0 <= successes <= n.")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive.")
    rate = successes / n
    den = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / den
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def metrics(p: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    p = _probabilities(p)
    y = _labels(y, len(p))
    sets = prediction_sets(p, threshold)
    covered = sets[np.arange(len(y)), y]
    sizes = sets.sum(axis=1)
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    ece = 0.0
    # Positive-class calibration in ten fixed bins; descriptive, bin-sensitive.
    bins = np.minimum((p * 10).astype(int), 9)
    for b in range(10):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    lo, hi = wilson_interval(int(covered.sum()), len(y))
    return {
        "n_test_specimens": len(y), "accuracy": float(((p >= .5) == y).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "ece_10_bins": float(ece), "coverage": float(covered.mean()),
        "coverage_wilson_low": lo, "coverage_wilson_high": hi,
        "mean_set_size": float(sizes.mean()), "empty_set_rate": float((sizes == 0).mean()),
        "singleton_rate": float((sizes == 1).mean()),
    }


def paired_brier_bootstrap(p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray,
                           seed: int = 0, n_bootstrap: int = 1000) -> dict:
    """Paired specimen resampling; negative difference favors model A.

    The percentile interval is conditional on fixed training/calibration, not a
    confidence interval for the full model-development process.
    """
    p_a, p_b = _probabilities(p_a), _probabilities(p_b)
    if p_a.shape != p_b.shape or n_bootstrap < 2:
        raise ValueError("Matching probabilities and at least two resamples required.")
    y = _labels(y, len(p_a))
    delta = (p_a - y) ** 2 - (p_b - y) ** 2
    rng = np.random.default_rng(seed)
    sampled = delta[rng.integers(0, len(y), size=(n_bootstrap, len(y)))].mean(axis=1)
    return {"difference": float(delta.mean()), "ci_low": float(np.quantile(sampled, .025)),
            "ci_high": float(np.quantile(sampled, .975)), "n_bootstrap": n_bootstrap,
            "scope": "paired test specimens; conditional on fixed fitted models"}
