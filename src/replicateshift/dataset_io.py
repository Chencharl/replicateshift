"""Strict CSV ingestion for independent specimens with technical repeats.

The input contract intentionally has no patient, family, or shared-batch fields.
A row is a technical measurement; a prediction and label belong to a specimen.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SpecimenDataset:
    """Padded sets of repeats, with all labels and IDs at specimen level."""

    X: np.ndarray
    mask: np.ndarray
    y: np.ndarray | None
    specimen_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    source_sha256: str

    @property
    def n_specimens(self) -> int:
        return len(self.specimen_ids)


def read_specimen_csv(
    path: str | Path,
    *,
    require_labels: bool = True,
    expected_features: tuple[str, ...] | list[str] | None = None,
) -> SpecimenDataset:
    """Read a long table, rejecting inconsistent labels and incomplete features.

    Required columns: specimen_id, replicate_id, and >=1 feature_* column.
    label is required for fitting and optional for prediction; if present, every
    row must have 0 or 1. Feature columns can be reordered, but their names must
    match an expected schema exactly. Identifiers remain strings (including
    leading zeros). Specimens and their repeats are sorted lexicographically.
    """
    source = Path(path)
    raw = source.read_bytes()
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV must use UTF-8 encoding") from error
    reader = csv.DictReader(io.StringIO(decoded, newline=""), strict=True)
    try:
        names = reader.fieldnames
        rows = list(reader)
    except csv.Error as error:
        raise ValueError(f"Malformed CSV: {error}") from error
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("CSV must have unique, nonempty column names")
    missing = {"specimen_id", "replicate_id"} - set(names)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    features = tuple(name for name in names if name.startswith("feature_") and len(name) > 8)
    unsupported = set(names) - {"specimen_id", "replicate_id", "label", *features}
    if unsupported:
        raise ValueError(f"Unsupported columns: {sorted(unsupported)}; only specimen_id, "
                         "replicate_id, label, and feature_* are supported")
    if not features:
        raise ValueError("At least one numeric feature_* column is required")
    if expected_features is not None:
        expected = tuple(expected_features)
        if not expected or len(set(expected)) != len(expected) or set(features) != set(expected):
            raise ValueError(f"Feature schema mismatch: expected {list(expected)}, found {list(features)}")
        features = expected
    has_labels = "label" in names
    if require_labels and not has_labels:
        raise ValueError("Fitting requires a label column with binary values 0 and 1")
    groups: dict[str, dict[str, list[float]]] = {}
    labels: dict[str, int] = {}
    for line, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV row {line} has the wrong number of fields")
        specimen_id, replicate_id = row["specimen_id"], row["replicate_id"]
        if (not specimen_id.strip() or not replicate_id.strip()
                or specimen_id != specimen_id.strip() or replicate_id != replicate_id.strip()):
            raise ValueError(f"CSV row {line}: identifiers must be nonempty without surrounding whitespace")
        if any(character in specimen_id + replicate_id for character in "\r\n\x00"):
            raise ValueError(f"CSV row {line}: identifiers cannot contain control characters")
        repeats = groups.setdefault(specimen_id, {})
        if replicate_id in repeats:
            raise ValueError(f"Duplicate specimen/replicate pair at CSV row {line}")
        try:
            values = [float(row[name]) for name in features]
        except (TypeError, ValueError) as error:
            raise ValueError(f"CSV row {line}: every feature must be numeric and nonmissing") from error
        if not np.isfinite(values).all():
            raise ValueError(f"CSV row {line}: features must contain finite values")
        repeats[replicate_id] = values
        if has_labels:
            if row["label"] not in ("0", "1"):
                raise ValueError(f"CSV row {line}: label must be exactly 0 or 1")
            label = int(row["label"])
            if specimen_id in labels and labels[specimen_id] != label:
                raise ValueError(f"Contradictory labels for specimen {specimen_id!r}")
            labels[specimen_id] = label
    if not groups:
        raise ValueError("CSV must contain at least one measurement row")
    ids = tuple(sorted(groups))
    X = np.zeros((len(ids), max(map(len, groups.values())), len(features)), dtype=np.float64)
    mask = np.zeros(X.shape[:2], dtype=bool)
    for i, specimen_id in enumerate(ids):
        repeats = groups[specimen_id]
        values = [repeats[replicate_id] for replicate_id in sorted(repeats)]
        X[i, :len(values)] = values
        mask[i, :len(values)] = True
    y = np.array([labels[key] for key in ids], dtype=np.int64) if has_labels else None
    return SpecimenDataset(X, mask, y, ids, features, hashlib.sha256(raw).hexdigest())


def split_specimens(n_specimens: int, seed: int) -> dict[str, np.ndarray]:
    """50/15/15/20% random train/selection/calibration/test partition.

    Splitting is independent of labels and uses largest-remainder allocation.
    The 60-specimen minimum provides at least nine calibration specimens; it
    does not establish scientific adequacy or guarantee informative sets.
    """
    if (isinstance(n_specimens, (bool, np.bool_))
            or not isinstance(n_specimens, (int, np.integer)) or n_specimens < 60):
        raise ValueError("Fitting requires at least 60 independent specimens")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    target = n_specimens * np.array([.5, .15, .15, .2])
    counts = np.floor(target).astype(int)
    order = np.argsort(-(target - counts), kind="stable")
    counts[order[:n_specimens - counts.sum()]] += 1
    partitions = np.split(np.random.default_rng(seed).permutation(n_specimens), np.cumsum(counts)[:-1])
    return dict(zip(("train", "selection", "calibration", "test"), partitions))
