"""Small, explicitly synthetic benchmark. Run with --help for the CLI."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import tracemalloc

import numpy as np

from . import __version__
from .inference import conformal_threshold, metrics, paired_brier_bootstrap, prediction_sets
from .models import Standardizer, LogisticClassifier, MLPClassifier, NearestNeighborClassifier
from .simulation import Dataset, generate_dataset, split_by_specimen, split_by_row, aggregate_specimens


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def role_seeds(seed: int) -> dict[str, int]:
    """Independent spawned streams for data, splits, fitting, and resampling."""
    roles = ("simulation", "split", "logistic", "interaction_logistic", "mlp", "bootstrap", "leakage")
    return {name: int(child.generate_state(1)[0])
            for name, child in zip(roles, np.random.SeedSequence(seed).spawn(len(roles)))}


def source_hashes() -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feature_map(X: np.ndarray, model: str) -> np.ndarray:
    """All pairwise interactions give the neural model a nonlinear comparator.

    This deterministic map uses no labels or fitted statistics. Scaling follows
    the map and is fitted independently on each model's training matrix only.
    """
    if model != "interaction_logistic":
        return X
    left, right = np.triu_indices(X.shape[1], k=1)
    return np.column_stack((X, X[:, left] * X[:, right]))


def leakage_control(data: Dataset, seed: int) -> list[dict]:
    """Permutation negative control; no biological signal should remain.

    Row splitting predicts another repeat of often-seen specimens. Group splitting
    predicts unseen specimens. These are distinct targets, intentionally contrasted.
    """
    ids, first = np.unique(data.specimen_id, return_index=True)
    shuffled = np.random.default_rng(seed + 10000).permutation(data.y[first])
    lookup = dict(zip(ids, shuffled))
    permuted = Dataset(data.X, np.array([lookup[i] for i in data.specimen_id]),
                       data.specimen_id, data.replicate_id)
    output = []
    for name, splitter in (("row_split", split_by_row), ("specimen_split", split_by_specimen)):
        train, _, test = splitter(permuted, seed=seed)
        scaler = Standardizer().fit(data.X[train])
        model = NearestNeighborClassifier().fit(scaler.transform(data.X[train]), permuted.y[train])
        p = model.predict_proba(scaler.transform(data.X[test]))
        seen = np.isin(data.specimen_id[test], data.specimen_id[train])
        output.append({"seed": seed, "split": name, "model": "1_nearest_neighbor",
                       "n_test_rows": len(test), "accuracy_per_row": float(((p >= .5) == permuted.y[test]).mean()),
                       "test_rows_with_specimen_seen_in_training": float(seen.mean()),
                       "labels": "permuted across specimens; shared within specimen"})
    return output


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for model, shift in sorted({(r["model"], r["shift"]) for r in rows}):
        group = [r for r in rows if r["model"] == model and r["shift"] == shift]
        summary = {"model": model, "shift": shift, "n_seeds": len(group)}
        for key in ("accuracy", "brier", "coverage", "mean_set_size", "singleton_rate", "empty_set_rate"):
            values = np.array([r[key] for r in group])
            summary[key + "_mean"] = float(values.mean())
            summary[key + "_mcse"] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
        output.append(summary)
    return output


def report(config: dict, summary: list[dict], rows: list[dict], leakage: list[dict], comparisons: list[dict]) -> str:
    lines = ["# ReplicateShift: synthetic smoke study", "",
             "Status: executed prototype validation. This is not a powered simulation study or biological validation.", "",
             f"Seeds: {config['seeds']}; {config['specimens']} specimens per seed, {config['replicates']} technical repeats each. "
             f"Train/calibration/test: 60/20/20% by specimen. Target coverage: {1-config['alpha']:.0%}.", "",
             "## Specimen-level results", "",
             "Each entry is the arithmetic mean across seeds. Read per_seed_metrics.csv for every run, coverage intervals, and uncertainty diagnostics.", "",
             "| Model | Assay shift | Accuracy | Brier ↓ | Coverage | Mean set size | Singleton rate |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['model']} | {row['shift']:g} | {row['accuracy_mean']:.3f} | {row['brier_mean']:.3f} | "
                     f"{row['coverage_mean']:.3f} | {row['mean_set_size_mean']:.3f} | {row['singleton_rate_mean']:.3f} |")
    lines += ["", "Coverage must be read with set size: predicting both labels always covers the true label. "
              f"The {1-config['alpha']:.0%} guarantee is marginal over exchangeable calibration and new test specimens, conditional on independent training. "
              "It is not a per-specimen, subgroup, fixed-calibration, or shifted-distribution guarantee.", "",
              "## Leakage negative control", "",
              "Labels were permuted across specimens and repeated within each specimen. A 1-nearest-neighbor model tests whether "
              "specimen fingerprints can cross the split. These row-level accuracies describe different prediction tasks; they are not comparable to the table above.", "",
              "| Seed | Split | Accuracy per row | Test rows from training specimens |", "|---:|---|---:|---:|"]
    for row in leakage:
        lines.append(f"| {row['seed']} | {row['split']} | {row['accuracy_per_row']:.3f} | {row['test_rows_with_specimen_seen_in_training']:.3f} |")
    lines += ["", "## Paired model comparison", "",
              "Difference in test Brier score: MLP minus each baseline. Negative favors the MLP. "
              "95% percentile bootstrap intervals resample test specimens jointly for both models, conditional on the fitted models. "
              "They do not include retraining uncertainty or establish significance across the full benchmark.", "",
              "| Seed | Assay shift | Baseline | Brier difference | Conditional interval |", "|---:|---:|---|---:|---|"]
    for row in comparisons:
        lines.append(f"| {row['seed']} | {row['shift']:g} | {row['baseline']} | {row['difference']:.3f} | [{row['ci_low']:.3f}, {row['ci_high']:.3f}] |")
    lines += ["", "## Interpretation limits", "",
              "- The neural network, linear model, and conformal procedure are established methods. The project contribution is the reproducible experiment and its research extension.",
              "- Only synthetic features and labels were used. The simulator does not model RNA chemistry, actual mass spectra, or clinical outcomes.",
              "- The shift changes measured features with the latent specimen labels held fixed. It need not preserve P(Y|X).",
              "- Shift scenarios reuse paired latent test specimens. Do not treat scenarios as independent replicates.",
              f"- This run uses {len(config['seeds'])} seeds. Summary MCSE values are descriptive; the default three-seed run is a smoke study, and no superiority claim is made.",
              "- Wilson intervals describe independent test outcomes conditional on each fixed fit and calibration set; calibration variability is assessed only through repeated seeds.",
              "- The simulator has independent specimens and no shared random batch effect. Real batch dependence requires additional splitting and inferential design.",
              "- Runtime is measured for this smoke run. tracemalloc peaks are traced allocations, not total resident or device memory.", "",
              "## Reproduce", "", "```sh", config["reproduction_command"], "```", "",
              "Use a new output directory; the CLI refuses to overwrite existing results. "
              "manifest.json records configuration, source hashes, data/split hashes, software versions, time, and memory scope.", ""]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    if args.specimens < 60 or args.replicates < 2 or args.epochs < 1 or args.bootstrap < 2:
        raise ValueError("Require specimens >=60, replicates >=2, epochs >=1, bootstrap >=2.")
    if not 0 < args.alpha < 1 or not all(np.isfinite(args.shifts)):
        raise ValueError("Require alpha in (0,1) and finite shifts.")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be distinct nonnegative integers.")
    if len(set(args.shifts)) != len(args.shifts):
        raise ValueError("Shifts must be distinct.")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; choose a new --output directory.")
    out.mkdir(parents=True)
    config = vars(args).copy()
    config.pop("output")
    config["reproduction_command"] = (f"PYTHONPATH=src python -m replicateshift.benchmark --specimens {args.specimens} "
        f"--replicates {args.replicates} --epochs {args.epochs} --alpha {args.alpha} --bootstrap {args.bootstrap} "
        f"--seeds {' '.join(map(str,args.seeds))} --shifts {' '.join(map(str,args.shifts))} --output results/local-reproduction")
    # Save protocol before any fitting/evaluation. No hyperparameter selection on test/calibration.
    (out / "protocol.json").write_text(json.dumps(config, indent=2) + "\n")
    sources = source_hashes()
    started = time.perf_counter()
    tracemalloc.start()
    all_metrics, all_predictions, all_leakage, comparisons, provenance = [], [], [], [], []
    for seed in args.seeds:
        streams = role_seeds(seed)
        data = generate_dataset(n_specimens=args.specimens, n_replicates=args.replicates, seed=streams["simulation"])
        train, calibration, test = split_by_specimen(data, seed=streams["split"])
        xtrain, ytrain, train_ids = aggregate_specimens(data, train)
        xcal, ycal, cal_ids = aggregate_specimens(data, calibration)
        _, _, test_ids = aggregate_specimens(data, test)
        assert not (set(train_ids) & set(cal_ids) or set(train_ids) & set(test_ids) or set(cal_ids) & set(test_ids))
        models = {"logistic": LogisticClassifier(seed=streams["logistic"], epochs=args.epochs),
                  "interaction_logistic": LogisticClassifier(seed=streams["interaction_logistic"], epochs=args.epochs),
                  "mlp": MLPClassifier(seed=streams["mlp"], epochs=args.epochs)}
        scalers = {name: Standardizer().fit(feature_map(xtrain, name)) for name in models}
        thresholds, fit_seconds = {}, {}
        for name, model in models.items():
            tick = time.perf_counter()
            model.fit(scalers[name].transform(feature_map(xtrain, name)), ytrain)
            fit_seconds[name] = time.perf_counter() - tick
            thresholds[name] = conformal_threshold(model.predict_proba(scalers[name].transform(feature_map(xcal, name))), ycal, args.alpha)
        provenance.append({"seed": seed, "role_seeds": streams, "data_sha256": array_hash(data.X, data.y, data.specimen_id, data.replicate_id),
                           "train_specimen_ids": train_ids.tolist(), "calibration_specimen_ids": cal_ids.tolist(),
                           "test_specimen_ids": test_ids.tolist(), "split_sha256": array_hash(train, calibration, test)})
        for shift in args.shifts:
            shifted = generate_dataset(n_specimens=args.specimens, n_replicates=args.replicates, seed=streams["simulation"], shift=shift)
            xtest, ytest, ids = aggregate_specimens(shifted, test)
            probabilities = {}
            for name, model in models.items():
                p = model.predict_proba(scalers[name].transform(feature_map(xtest, name)))
                probabilities[name] = p
                q = thresholds[name]
                all_metrics.append({"seed": seed, "model": name, "shift": shift,
                                    "n_train_specimens": len(train_ids), "n_calibration_specimens": len(cal_ids),
                                    "threshold": q if np.isfinite(q) else "infinity", "fit_seconds": fit_seconds[name],
                                    **metrics(p, ytest, q)})
                sets = prediction_sets(p, q)
                all_predictions.extend({"seed": seed, "model": name, "shift": shift, "specimen_id": int(i),
                    "label": int(y), "p_label_1": float(prob), "include_0": int(s[0]), "include_1": int(s[1])}
                    for i, y, prob, s in zip(ids, ytest, p, sets))
            for baseline in ("logistic", "interaction_logistic"):
                comparisons.append({"seed": seed, "shift": shift, "baseline": baseline, **paired_brier_bootstrap(
                    probabilities["mlp"], probabilities[baseline], ytest, seed=streams["bootstrap"], n_bootstrap=args.bootstrap)})
        leakage = leakage_control(data, streams["leakage"])
        all_leakage.extend({**row, "seed": seed} for row in leakage)
        print(f"Completed seed {seed}: {len(train_ids)} train / {len(cal_ids)} calibration / {len(test_ids)} test specimens", flush=True)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    summary = summarize(all_metrics)
    for filename, rows in (("per_seed_metrics.csv", all_metrics), ("predictions.csv", all_predictions),
                           ("leakage_control.csv", all_leakage), ("paired_brier.csv", comparisons), ("summary.csv", summary)):
        write_csv(out / filename, rows)
    (out / "REPORT.md").write_text(report(config, summary, all_metrics, all_leakage, comparisons), encoding="utf-8")
    if source_hashes() != sources:
        raise RuntimeError("Source files changed during execution; results are not a completed reproducible run.")
    manifest = {"status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
                "project_version": __version__, "python": platform.python_version(), "numpy": np.__version__,
                "platform": platform.system() + " " + platform.machine(), "configuration": config,
                "elapsed_seconds": elapsed, "peak_traced_bytes": peak,
                "memory_scope": "tracemalloc allocations; not total resident or accelerator memory",
                "sources_sha256": sources, "datasets_and_splits": provenance,
                "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Saved {out}; {elapsed:.2f} seconds; {peak / 1024**2:.2f} MiB peak traced allocations", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specimens", type=int, default=600)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 43])
    parser.add_argument("--shifts", type=float, nargs="+", default=[0., 1.5])
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--alpha", type=float, default=.1)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output", default="results/local-demo")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
