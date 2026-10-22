"""Known-truth repeated-measurement experiments using entirely synthetic data.

This is an omics-inspired statistical simulator, not LC-MS chemistry or a model
of biological truth. Specimens are IID; the dependence is between technical
replicates of the same specimen. There is no shared random batch effect.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Dataset:
    """Row-level measurements, a specimen-level binary label, and group IDs.

    Specimen IDs need not be consecutive. Replicate IDs must identify distinct
    rows within a specimen. Arrays are not made immutable by the frozen wrapper.
    """

    X: NDArray[np.float64]
    y: NDArray[np.int64]
    specimen_id: NDArray[np.int64]
    replicate_id: NDArray[np.int64]

    def __post_init__(self) -> None:
        X = np.asarray(self.X)
        y = np.asarray(self.y)
        specimen_id = np.asarray(self.specimen_id)
        replicate_id = np.asarray(self.replicate_id)
        if X.ndim != 2 or min(X.shape) < 1:
            raise ValueError("X must be a nonempty two-dimensional array")
        if not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X):
            raise ValueError("X must contain real numeric measurements")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite measurements")
        for name, values in (("y", y), ("specimen_id", specimen_id),
                             ("replicate_id", replicate_id)):
            if values.ndim != 1 or len(values) != len(X):
                raise ValueError(f"{name} must be one-dimensional with one value per row")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only binary labels 0 and 1")
        for name, values in (("specimen_id", specimen_id),
                             ("replicate_id", replicate_id)):
            if not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{name} must contain integer identifiers")
        pairs = np.column_stack((specimen_id, replicate_id))
        if len(np.unique(pairs, axis=0)) != len(X):
            raise ValueError("replicate IDs must be unique within each specimen")
        _, inverse = np.unique(specimen_id, return_inverse=True)
        counts = np.bincount(inverse)
        positives = np.bincount(inverse, weights=y.astype(float))
        if np.any((positives != 0) & (positives != counts)):
            raise ValueError("all rows of a specimen must have the same label")
        object.__setattr__(self, "X", X.astype(np.float64, copy=False))
        object.__setattr__(self, "y", y.astype(np.int64, copy=False))
        object.__setattr__(self, "specimen_id", specimen_id.astype(np.int64, copy=False))
        object.__setattr__(self, "replicate_id", replicate_id.astype(np.int64, copy=False))


def _positive_integer(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _rng(seed: int) -> np.random.Generator:
    _positive_integer(seed, "seed", minimum=0)
    return np.random.default_rng(seed)


def generate_dataset(
    n_specimens: int = 400,
    n_replicates: int = 3,
    n_features: int = 12,
    seed: int = 0,
    shift: float = 0.0,
    noise: float = 0.35,
) -> Dataset:
    """Draw independent specimens with correlated technical measurements.

    Each specimen has three independent standard-normal latent signals z. Its
    binary label is drawn once with log odds ``2.5*z[0]*z[1] + 1.2*z[2]``.
    Features 3 onward are independent normal nuisance fingerprints (SD 1.5),
    shared within a specimen and independent of its label-generating mechanism.
    Each row receives independent normal measurement noise with SD ``noise``.

    An assay shift adds ``shift`` to measured feature 0, attenuates measured
    signal 1 by ``1 / (1 + abs(shift))``, and adds ``0.5*shift`` to feature 2,
    before noise is added. It does not change the latent label mechanism.
    Identical seeds and dimensions preserve specimen labels and measurement
    noise across shift values, enabling a paired sensitivity experiment. Such
    paired datasets must not be treated as independent cohorts.
    """
    _positive_integer(n_specimens, "n_specimens")
    _positive_integer(n_replicates, "n_replicates")
    _positive_integer(n_features, "n_features", minimum=3)
    if not isinstance(shift, Real) or not np.isfinite(shift):
        raise ValueError("shift must be a finite scalar")
    if not isinstance(noise, Real) or not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be a finite nonnegative scalar")
    rng = _rng(seed)
    latent = rng.normal(size=(n_specimens, 3))
    log_odds = 2.5 * latent[:, 0] * latent[:, 1] + 1.2 * latent[:, 2]
    probability = np.exp(-np.logaddexp(0.0, -log_odds))
    labels = rng.binomial(1, probability)
    nuisance = rng.normal(scale=1.5, size=(n_specimens, n_features - 3))
    signal = latent.copy()
    signal[:, 0] += shift
    signal[:, 1] /= 1.0 + abs(shift)
    signal[:, 2] += 0.5 * shift
    specimen_features = np.column_stack((signal, nuisance))
    X = np.repeat(specimen_features, n_replicates, axis=0)
    X += rng.normal(scale=noise, size=X.shape)
    return Dataset(
        X=X,
        y=np.repeat(labels, n_replicates),
        specimen_id=np.repeat(np.arange(n_specimens), n_replicates),
        replicate_id=np.tile(np.arange(n_replicates), n_specimens),
    )


def _split_units(
    units: NDArray[np.int64], seed: int, fractions: tuple[float, float, float]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    fractions_array = np.asarray(fractions, dtype=float)
    if (fractions_array.shape != (3,) or not np.all(np.isfinite(fractions_array))
            or np.any(fractions_array <= 0)
            or not np.isclose(fractions_array.sum(), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("fractions must be three positive finite numbers summing to one")
    target = len(units) * fractions_array
    counts = np.floor(target).astype(int)
    remainder_order = np.argsort(-(target - counts), kind="stable")
    counts[remainder_order[:len(units) - counts.sum()]] += 1
    if np.any(counts == 0):
        raise ValueError("too few units for three nonempty splits at the requested fractions")
    shuffled = _rng(seed).permutation(units)
    train_end, cal_end = np.cumsum(counts)[:2]
    return shuffled[:train_end], shuffled[train_end:cal_end], shuffled[cal_end:]


def split_by_specimen(
    data: Dataset,
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Return train/calibration/test row indices without specimen overlap.

    The specimen permutation is independent of labels; no stratification is
    applied. Sizes use largest-remainder allocation; empty partitions raise.
    """
    groups = _split_units(np.unique(data.specimen_id), seed, fractions)
    return tuple(np.flatnonzero(np.isin(data.specimen_id, ids)) for ids in groups)


def split_by_row(
    data: Dataset,
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Negative control: randomly split rows, allowing specimen leakage.

    Calibration and test rows need not be independent of training rows. Results
    from this split do not establish performance on previously unseen specimens.
    """
    return _split_units(np.arange(len(data.y)), seed, fractions)


def aggregate_specimens(
    data: Dataset, indices: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Average supplied rows within specimens, returning sorted specimen IDs.

    A subset of a specimen's replicates is allowed and only supplied rows enter
    its mean. Consequently, aggregating separate row-level partitions does not
    remove specimen leakage. Duplicate, empty, or out-of-range indices raise.
    """
    rows = np.asarray(indices)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or len(rows) == 0:
        raise ValueError("indices must be a nonempty one-dimensional integer array")
    if np.any(rows < 0) or np.any(rows >= len(data.y)):
        raise ValueError("indices are out of bounds")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("indices must not contain duplicate rows")
    ids, first, inverse, counts = np.unique(
        data.specimen_id[rows], return_index=True, return_inverse=True, return_counts=True
    )
    Xmean = np.zeros((len(ids), data.X.shape[1]), dtype=float)
    np.add.at(Xmean, inverse, data.X[rows])
    Xmean /= counts[:, None]
    y = data.y[rows[first]]
    if not np.array_equal(data.y[rows], y[inverse]):
        raise ValueError("all selected rows of a specimen must have the same label")
    return Xmean, y, ids
