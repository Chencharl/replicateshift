"""Tests of allocation accounting, paired inference, and frozen checkpoints."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import replicateshift.allocation as allocation
from replicateshift.allocation import (
    MODELS,
    draw_pool,
    pool_subset,
    run,
    summarize_runs,
    summary_features,
    validate_config,
)
from replicateshift.benchmark import array_hash
from replicateshift.models import LogisticClassifier, MLPClassifier


def small_config():
    return {
        "budget": 60, "repeats": [1, 3], "noise_levels": [.3, .9],
        "seeds": [17, 29], "test_specimens": 12, "features": 3,
        "alpha": .2, "epochs": 3, "logistic_learning_rate": .03,
        "set_learning_rate": .01, "l2": .003, "set_hidden": [4, 3],
        "set_head_width": 3, "max_seconds": 30.,
    }


def illustrative_metrics():
    """Three complete paired runs with analytically simple Brier differences."""
    briers = {"deep_set": [.2, .4, .6], "mean_interaction": [.1, .3, .3],
              "mean_sd_interaction": [.2, .4, .6]}
    rows = []
    for model in MODELS:
        for seed, brier in zip([5, 9, 13], briers[model]):
            rows.append({"seed": seed, "noise": .5, "repeats": 3, "model": model,
                         "development_specimens": 20, "development_assays": 60,
                         "n_test_specimens": 10, "test_assays": 30,
                         "brier": brier, "accuracy": .7, "log_loss": .8,
                         "coverage": .9, "mean_set_size": 1.5,
                         "empty_set_rate": 0., "singleton_rate": .5})
    return rows


class AllocationHelperTests(unittest.TestCase):
    def test_allocation_prefixes_are_nested_and_specimen_roles_never_change(self):
        roles = ("train", "validation", "calibration", "test")
        children = np.random.SeedSequence(71).spawn(len(roles))
        seeds = {role: int(child.generate_state(1)[0]) for role, child in zip(roles, children)}
        pools = {role: draw_pool(18, 3, 4, seeds[role], role) for role in roles}
        all_ids = [item for pool in pools.values() for item in pool["ids"]]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(len(seeds), len(set(seeds.values())))
        for role, pool in pools.items():
            with self.subTest(role=role):
                many_specimens, many_mask, many_y = pool_subset(pool, 18, 1, .5)
                more_repeats, repeat_mask, repeat_y = pool_subset(pool, 6, 3, .5)
                np.testing.assert_array_equal(many_specimens[:6, :1], more_repeats[:, :1])
                np.testing.assert_array_equal(many_y[:6], repeat_y)
                self.assertTrue(many_mask.all() and repeat_mask.all())
                self.assertEqual(pool["ids"][:6], [f"{role}-{i:05}" for i in range(6)])
                replay = draw_pool(18, 3, 4, seeds[role], role)
                for field in ("base", "noise", "y"):
                    np.testing.assert_array_equal(pool[field], replay[field])
        self.assertFalse(np.array_equal(pools["train"]["base"], pools["test"]["base"]))

    def test_noise_scenarios_scale_the_same_draw_without_changing_labels(self):
        pool = draw_pool(8, 3, 4, seed=11, role="train")
        low, _, labels_low = pool_subset(pool, 7, 3, .4)
        high, _, labels_high = pool_subset(pool, 7, 3, 1.2)
        no_noise, _, labels_zero = pool_subset(pool, 7, 3, 0.)
        base = np.broadcast_to(pool["base"][:7, None, :], low.shape)
        np.testing.assert_array_equal(no_noise, base)
        np.testing.assert_allclose(high-base, 3*(low-base), atol=2e-15)
        np.testing.assert_array_equal(labels_low, labels_high)
        np.testing.assert_array_equal(labels_low, labels_zero)

    def test_budget_divisibility_and_minimum_calibration_size(self):
        config = validate_config(small_config())
        for repeats in config["repeats"]:
            counts = [3*config["budget"]//(5*repeats),
                      config["budget"]//(5*repeats), config["budget"]//(5*repeats)]
            self.assertEqual(sum(counts)*repeats, config["budget"])
            self.assertEqual(sum(counts), config["budget"]//repeats)
            self.assertGreaterEqual(min(counts), 2)
        for changed in ({"budget": 61}, {"budget": True}, {"repeats": [5]},
                        {"repeats": [12]}, {"repeats": [1, 1]}, {"seeds": [5, 5]},
                        {"noise_levels": [.3, .3]}, {"alpha": 1.}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_config({**small_config(), **changed})

    def test_sd_ablation_retains_all_mean_interactions_and_uses_population_sd(self):
        X = np.array([[[1., 2., 3.], [3., 6., 9.]],
                      [[4., 6., 8.], [4., 6., 8.]]])
        baseline = summary_features(X, "mean_interaction")
        expanded = summary_features(X, "mean_sd_interaction")
        np.testing.assert_array_equal(baseline, [[2., 4., 6., 8., 12., 24.],
                                               [4., 6., 8., 24., 32., 48.]])
        np.testing.assert_array_equal(expanded[:, :6], baseline)
        np.testing.assert_array_equal(expanded[:, 6:], [[1., 2., 3.], [0., 0., 0.]])
        one_repeat = summary_features(X[:, :1], "mean_sd_interaction")
        np.testing.assert_array_equal(one_repeat[:, -3:], 0.)


class PairedSummaryTests(unittest.TestCase):
    def test_summary_and_paired_mcse_match_hand_calculation(self):
        # Reordering rows must not turn paired comparisons into position pairing.
        rows = illustrative_metrics()
        rows = [rows[i] for i in [8, 1, 4, 0, 6, 3, 2, 7, 5]]
        summaries, paired = summarize_runs(rows)
        deep = next(row for row in summaries if row["model"] == "deep_set")
        self.assertEqual(deep["n_seeds"], 3)
        self.assertAlmostEqual(deep["brier_mean"], .4)
        self.assertAlmostEqual(deep["brier_mcse"], .2/np.sqrt(3))
        difference = next(row for row in paired if row["baseline"] == "mean_interaction")
        # Differences [.1, .1, .3] have mean 1/6 and SE 1/15, whereas an
        # unpaired calculation would use two unrelated variance components.
        self.assertAlmostEqual(difference["deep_set_minus_baseline_brier"], 1/6)
        self.assertAlmostEqual(difference["mcse_paired_difference"], 1/15)
        identical = next(row for row in paired if row["baseline"] == "mean_sd_interaction")
        self.assertEqual(identical["deep_set_minus_baseline_brier"], 0.)
        self.assertEqual(identical["mcse_paired_difference"], 0.)

    def test_single_run_has_no_estimable_mcse(self):
        rows = [row for row in illustrative_metrics() if row["seed"] == 5]
        summaries, paired = summarize_runs(rows)
        self.assertTrue(all(row["brier_mcse"] is None for row in summaries))
        self.assertTrue(all(row["mcse_paired_difference"] is None for row in paired))

    def test_duplicate_missing_or_unpaired_run_records_are_rejected(self):
        rows = illustrative_metrics()
        cases = [[], rows + [dict(rows[0])], rows[:-1],
                 [row for row in rows if row["model"] != "deep_set"]]
        for records in cases:
            with self.subTest(length=len(records)), self.assertRaises(ValueError):
                summarize_runs(records)


class AllocationArtifactTests(unittest.TestCase):
    def test_small_run_hashes_predictions_costs_and_overwrite_boundary(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"pilot"
            run(config, output)
            manifest = json.loads((output/"manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["configuration"], config)
            self.assertEqual(json.loads((output/"protocol.json").read_text()), config)
            hashed_outputs = manifest["output_sha256"]
            self.assertEqual(set(hashed_outputs), {p.name for p in output.iterdir()}-{"manifest.json"})
            for name, expected in hashed_outputs.items():
                self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(), expected)
            for name, expected in manifest["source_sha256"].items():
                source = Path(allocation.__file__).parent/name
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), expected)
            for provenance in manifest["role_pool_provenance"]:
                all_ids = []
                for role, recorded in provenance["pools"].items():
                    regenerated = draw_pool(recorded["n_specimens"], max(config["repeats"]),
                                            config["features"], provenance["role_seeds"][role], role)
                    self.assertEqual(recorded["ids"], regenerated["ids"])
                    self.assertEqual(recorded["base_label_noise_sha256"],
                                     array_hash(regenerated["base"], regenerated["y"], regenerated["noise"]))
                    all_ids.extend(recorded["ids"])
                self.assertEqual(len(all_ids), len(set(all_ids)))
            with (output/"per_run_metrics.csv").open(newline="") as handle:
                metric_rows = list(csv.DictReader(handle))
            with gzip.open(output/"predictions.csv.gz", "rt", newline="") as handle:
                predictions = list(csv.DictReader(handle))
            number_conditions = len(config["seeds"])*len(config["noise_levels"])*len(config["repeats"])*len(MODELS)
            self.assertEqual(len(metric_rows), number_conditions)
            self.assertEqual(len(predictions), number_conditions*config["test_specimens"])
            for row in metric_rows:
                repeats = int(row["repeats"])
                train, validation, calibration = (int(row[key]) for key in ("n_train", "n_validation", "n_calibration"))
                self.assertEqual((train*repeats, validation*repeats, calibration*repeats), (36, 12, 12))
                self.assertEqual(int(row["development_assays"]), config["budget"])
                self.assertEqual(int(row["development_specimens"]), train+validation+calibration)
                self.assertEqual(int(row["test_assays"]), config["test_specimens"]*repeats)
                self.assertTrue(1 <= int(row["best_epoch"]) <= config["epochs"])
                same_condition = [p for p in predictions if all(p[key] == row[key]
                                  for key in ("seed", "noise", "repeats", "model"))]
                self.assertEqual(len(same_condition), config["test_specimens"])
                p = np.array([float(item["p_label_1"]) for item in same_condition])
                y = np.array([int(item["label"]) for item in same_condition])
                coverage = np.mean([int(item["include_"+item["label"]]) for item in same_condition])
                self.assertAlmostEqual(float(row["brier"]), np.mean((p-y)**2))
                self.assertAlmostEqual(float(row["coverage"]), coverage)
            prior_manifest = (output/"manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                run(config, output)
            self.assertEqual((output/"manifest.json").read_bytes(), prior_manifest)


class BaselineCheckpointTests(unittest.TestCase):
    def test_logistic_and_mlp_checkpoints_replay_and_reset(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(30, 2))
        y = (X[:, 0] > 0).astype(float)
        for factory, extra in ((LogisticClassifier, {}), (MLPClassifier, {"hidden": (5, 4)})):
            with self.subTest(model=factory.__name__):
                options = dict(seed=2, epochs=30, learning_rate=.03, l2=.07, **extra)
                selected = factory(**options).fit(X, y, validation=(X, 1-y))
                final = factory(**options).fit(X, y)
                np.testing.assert_array_equal(selected.loss_history_, final.loss_history_)
                self.assertEqual(selected.best_epoch_, np.argmin(selected.validation_loss_history_)+1)
                self.assertLess(selected.best_epoch_, options["epochs"])
                replay = factory(**{**options, "epochs": selected.best_epoch_}).fit(X, y)
                for name in selected.parameters_:
                    np.testing.assert_array_equal(selected.parameters_[name], replay.parameters_[name])
                p = selected.predict_proba(X)
                bce = -np.mean((1-y)*np.log(p) + y*np.log1p(-p))
                self.assertAlmostEqual(bce, min(selected.validation_loss_history_), places=13)
                penalized, _ = selected.loss_and_gradients(X, 1-y)
                self.assertGreater(penalized, bce)
                selected.fit(X, y)
                self.assertEqual(selected.validation_loss_history_, [])
                self.assertEqual(selected.best_epoch_, options["epochs"])
                np.testing.assert_array_equal(selected.predict_proba(X), final.predict_proba(X))

    def test_malformed_validation_fails_without_accidental_index_errors(self):
        X, y = np.array([[0., 1.], [1., 0.]]), np.array([0., 1.])
        for factory in (LogisticClassifier, MLPClassifier):
            for validation in ((), (X,), (X, y, y), (X[:, :1], y), (X, y[:, None])):
                with self.subTest(model=factory.__name__, length=len(validation)), self.assertRaises(ValueError):
                    factory(epochs=1).fit(X, y, validation=validation)


if __name__ == "__main__":
    unittest.main()
