"""Local CSV -> repeat-aware model -> specimen prediction-set workflow.

No input data leave the local process. Calibration is frozen after validation
checkpoint selection. Prediction accepts only new specimen IDs; IDs alone cannot
establish independence or detect a renamed patient/batch shared across files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import numpy as np

from .dataset_io import read_specimen_csv, split_specimens
from .inference import conformal_threshold, metrics, prediction_sets
from .set_model import DeepSetClassifier, SetStandardizer
from .simulation import generate_dataset
from . import __version__

BUNDLE_SCHEMA_VERSION = 1


def _seed_roles(seed: int) -> dict[str, int]:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    children = np.random.SeedSequence(seed).spawn(2)
    return {name: int(child.generate_state(1)[0])
            for name, child in zip(("split", "model"), children)}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_predictions(path: Path, ids: tuple[str, ...] | list[str], probabilities: np.ndarray,
                       threshold: float, labels: np.ndarray | None) -> None:
    sets = prediction_sets(probabilities, threshold)
    with path.open("x", newline="", encoding="utf-8") as handle:
        columns = ["specimen_id", "p_label1", "include0", "include1"]
        if labels is not None:
            columns.append("label")
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index, specimen_id in enumerate(ids):
            row: list[Any] = [specimen_id, float(probabilities[index]), int(sets[index, 0]), int(sets[index, 1])]
            if labels is not None:
                row.append(int(labels[index]))
            writer.writerow(row)


def _write_bundle(directory: Path, model: DeepSetClassifier, scaler: SetStandardizer,
                  metadata: dict[str, Any]) -> None:
    state = model.get_state()
    scaler_state = scaler.get_state()
    parameters = state.pop("parameters")
    arrays = {"parameter_" + key: value for key, value in parameters.items()}
    arrays.update({"scaler_mean": scaler_state["mean"], "scaler_scale": scaler_state["scale"]})
    np.savez_compressed(directory / "weights.npz", **arrays)
    metadata.update({
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "model_type": "DeepSetClassifier",
        "model_state": state,
        "scaler_schema_version": scaler_state["schema_version"],
        "weights_sha256": hashlib.sha256((directory / "weights.npz").read_bytes()).hexdigest(),
    })
    _write_json(directory / "model.json", metadata)


def _publish_directory(staged: Path, output: Path) -> None:
    # Reserve an absent destination first. os.rename alone may overwrite an
    # existing empty directory on POSIX. The reservation is ours to replace.
    output.mkdir(exist_ok=False)
    try:
        staged.replace(output)
    except OSError:
        if output.is_dir() and not any(output.iterdir()):
            output.rmdir()
        raise


def load_model(directory: str | Path) -> tuple[DeepSetClassifier, SetStandardizer, dict[str, Any], float]:
    """Restore arrays without pickle and validate version, shapes, and checksum."""
    directory = Path(directory)
    metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    if (not isinstance(metadata, dict)
            or type(metadata.get("bundle_schema_version")) is not int
            or metadata.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION
            or metadata.get("model_type") != "DeepSetClassifier"):
        raise ValueError("Unsupported saved model schema or model type")
    if not isinstance(metadata.get("model_state"), dict):
        raise ValueError("Saved model must contain a model_state object")
    payload = directory / "weights.npz"
    if hashlib.sha256(payload.read_bytes()).hexdigest() != metadata.get("weights_sha256"):
        raise ValueError("Saved model weights checksum mismatch")
    with np.load(payload, allow_pickle=False) as arrays:
        expected = {"parameter_" + key for key in ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4")}
        expected |= {"scaler_mean", "scaler_scale"}
        if set(arrays.files) != expected:
            raise ValueError("Unexpected or missing saved model arrays")
        state = dict(metadata["model_state"])
        state["parameters"] = {name.removeprefix("parameter_"): arrays[name].copy()
                               for name in expected if name.startswith("parameter_")}
        model = DeepSetClassifier.from_state(state)
        scaler = SetStandardizer.from_state({"schema_version": metadata.get("scaler_schema_version"),
                                             "mean": arrays["scaler_mean"].copy(),
                                             "scale": arrays["scaler_scale"].copy()})
    features = metadata.get("feature_names")
    if (not isinstance(features, list) or len(features) != model.n_features_in_
            or any(not isinstance(name, str) or not name.startswith("feature_") or len(name) < 9 for name in features)
            or len(set(features)) != len(features)
            or scaler.n_features_in_ != model.n_features_in_):
        raise ValueError("Invalid saved feature schema")
    raw_threshold = metadata.get("conformal_threshold")
    if raw_threshold == "+inf":
        threshold = math.inf
    elif (isinstance(raw_threshold, (float, int)) and not isinstance(raw_threshold, bool)
            and math.isfinite(raw_threshold) and 0 <= raw_threshold <= 1):
        threshold = float(raw_threshold)
    else:
        raise ValueError("Invalid saved conformal threshold")
    alpha = metadata.get("alpha")
    if (not isinstance(alpha, (float, int)) or isinstance(alpha, bool)
            or not math.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("Invalid saved conformal settings")
    partitions = metadata.get("split_specimen_ids")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "selection", "calibration", "test"}:
        raise ValueError("Invalid saved specimen partition")
    if any(not isinstance(group, list) or not group for group in partitions.values()):
        raise ValueError("Saved specimen partitions must be nonempty lists")
    all_ids = [item for group in partitions.values() for item in group]
    if (any(not isinstance(item, str) or not item for item in all_ids)
            or len(all_ids) != len(set(all_ids)) or any(not group for group in partitions.values())):
        raise ValueError("Saved specimen partitions must be nonempty and disjoint")
    return model, scaler, metadata, threshold


def fit_dataset(input_path: str | Path, output: str | Path, *, epochs: int = 120,
                seed: int = 2026, alpha: float = .1,
                hidden: tuple[int, int] = (16, 12), head_width: int = 12,
                learning_rate: float = .003, l2: float = .003) -> dict[str, Any]:
    """Fit once, select by validation BCE, calibrate once, and evaluate holdout.

    The output directory must not exist. It contains a portable model bundle,
    the full specimen-role manifest, holdout predictions, and evaluation metrics.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    data = read_specimen_csv(input_path, require_labels=True)
    roles = _seed_roles(seed)
    split = split_specimens(data.n_specimens, roles["split"])
    train, selection, calibration, test = (split[key] for key in ("train", "selection", "calibration", "test"))
    assert data.y is not None
    scaler = SetStandardizer().fit(data.X[train], data.mask[train])
    X = scaler.transform(data.X, data.mask)
    model = DeepSetClassifier(seed=roles["model"], epochs=epochs, hidden=hidden,
                              head_width=head_width, learning_rate=learning_rate, l2=l2)
    model.fit(X[train], data.mask[train], data.y[train],
              validation=(X[selection], data.mask[selection], data.y[selection]))
    threshold = conformal_threshold(model.predict_proba(X[calibration], data.mask[calibration]),
                                    data.y[calibration], alpha)
    probabilities = model.predict_proba(X[test], data.mask[test])
    evaluation = metrics(probabilities, data.y[test], threshold)
    evaluation.update({"scope": "held-out independent specimens, conditional on fitted model and calibration",
                       "nominal_coverage": 1 - alpha, "best_epoch": model.best_epoch_})
    metadata: dict[str, Any] = {
        "input_filename": Path(input_path).name,
        "source_sha256": data.source_sha256,
        "feature_names": list(data.feature_names),
        "label_mapping": {"0": 0, "1": 1},
        "alpha": float(alpha),
        "conformal_threshold": "+inf" if math.isinf(threshold) else threshold,
        "seed": int(seed), "role_seeds": roles,
        "split_method": "label-independent specimen permutation; 50/15/15/20 largest-remainder allocation",
        "split_specimen_ids": {key: [data.specimen_ids[index] for index in rows] for key, rows in split.items()},
        "split_label_counts": {key: {str(label): int((data.y[rows] == label).sum()) for label in (0, 1)}
                               for key, rows in split.items()},
        "n_specimens": data.n_specimens,
        "n_measurements": int(data.mask.sum()),
        "repeat_count_min": int(data.mask.sum(axis=1).min()),
        "repeat_count_max": int(data.mask.sum(axis=1).max()),
        "assumptions": ["specimens are independent; repeats within each specimen may be dependent",
                        "calibration and future specimens are exchangeable as whole repeat sets",
                        "no tuning with calibration or test labels; no patient/family/shared-batch hierarchy"],
        "software": {"replicateshift_version": __version__, "python_version": platform.python_version(),
                     "numpy_version": np.__version__, "model_format": "JSON and NPZ; no pickle"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".replicateshift-", dir=output.parent) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        _write_bundle(staged, model, scaler, metadata)
        _write_predictions(staged / "holdout_predictions.csv", [data.specimen_ids[index] for index in test],
                           probabilities, threshold, data.y[test])
        _write_json(staged / "holdout_metrics.json", evaluation)
        _publish_directory(staged, output)
    return evaluation


def predict_dataset(model_path: str | Path, input_path: str | Path,
                    output: str | Path) -> dict[str, Any]:
    """Write prediction sets for fresh specimens without refitting/calibration."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    model, scaler, metadata, threshold = load_model(model_path)
    data = read_specimen_csv(input_path, require_labels=False, expected_features=metadata["feature_names"])
    known_ids = {value for group in metadata["split_specimen_ids"].values() for value in group}
    overlap = known_ids.intersection(data.specimen_ids)
    if overlap:
        raise ValueError(f"Prediction input contains {len(overlap)} previously used specimen IDs; "
                         "provide new independent specimens, not renamed repeats")
    X = scaler.transform(data.X, data.mask)
    probabilities = model.predict_proba(X, data.mask)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".replicateshift-predict-", dir=output.parent) as temporary:
        staged = Path(temporary) / "predictions.csv"
        _write_predictions(staged, data.specimen_ids, probabilities, threshold, data.y)
        # Linking a complete local file is atomic and refuses existing paths.
        os.link(staged, output)
    result: dict[str, Any] = {"n_specimens": data.n_specimens, "output": str(output),
                              "prediction_source_sha256": data.source_sha256}
    if data.y is not None:
        result["metrics"] = metrics(probabilities, data.y, threshold)
    return result


def generate_example(output: str | Path, *, specimens: int = 120, replicates: int = 3,
                     seed: int = 2026) -> Path:
    """Create public synthetic rows with three omics-inspired numeric features."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    data = generate_dataset(n_specimens=specimens, n_replicates=replicates, n_features=3, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["specimen_id", "replicate_id", "label", "feature_0", "feature_1", "feature_2"])
        for i in range(len(data.y)):
            writer.writerow([f"synthetic_{seed}_{data.specimen_id[i]:06d}", f"r{data.replicate_id[i]}",
                             int(data.y[i]), *data.X[i]])
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit, select, calibrate, and evaluate specimen CSV")
    fit.add_argument("input", type=Path)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=120)
    fit.add_argument("--seed", type=int, default=2026)
    fit.add_argument("--alpha", type=float, default=.1)
    predict = commands.add_parser("predict", help="predict sets for new independent specimen CSV")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    example = commands.add_parser("generate-example", help="write an entirely synthetic assay CSV")
    example.add_argument("--output", type=Path, required=True)
    example.add_argument("--specimens", type=int, default=120)
    example.add_argument("--replicates", type=int, default=3)
    example.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        if args.command == "fit":
            result = fit_dataset(args.input, args.output, epochs=args.epochs, seed=args.seed, alpha=args.alpha)
        elif args.command == "predict":
            result = predict_dataset(args.model, args.input, args.output)
        else:
            result = {"output": str(generate_example(args.output, specimens=args.specimens,
                                                      replicates=args.replicates, seed=args.seed))}
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"replicateshift: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
