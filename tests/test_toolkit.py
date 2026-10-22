import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from replicateshift.dataset_io import read_specimen_csv, split_specimens
from replicateshift.inference import conformal_threshold
from replicateshift.toolkit import fit_dataset, generate_example, load_model, predict_dataset


class DatasetIOTests(unittest.TestCase):
    def test_string_ids_variable_repeats_and_named_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.csv"
            path.write_text("specimen_id,replicate_id,label,feature_b,feature_a\n"
                            "B,r2,1,4,3\n001,r1,0,6,5\nB,r1,1,2,1\n")
            data = read_specimen_csv(path, expected_features=("feature_a", "feature_b"))
            self.assertEqual(data.specimen_ids, ("001", "B"))
            self.assertEqual(data.feature_names, ("feature_a", "feature_b"))
            np.testing.assert_array_equal(data.mask, [[True, False], [True, True]])
            np.testing.assert_array_equal(data.X, [[[5, 6], [0, 0]], [[1, 2], [3, 4]]])
            np.testing.assert_array_equal(data.y, [0, 1])
            self.assertEqual(data.source_sha256, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_rejects_invalid_csv(self):
        header = "specimen_id,replicate_id,label,feature_x\n"
        cases = [
            (header + "a,r1,0,1\na,r1,0,2\n", "Duplicate"),
            (header + "a,r1,0,1\na,r2,1,2\n", "Contradictory"),
            (header + "a,r1,0,nan\n", "finite"),
            (header + "a,r1,0,inf\n", "finite"),
            (header + "a,r1,0,\n", "numeric"),
            (header + "a,r1,2,3\n", "label"),
            (header + "a,r1,0\n", "number of fields"),
            (header + "a,r1,0,1,extra\n", "number of fields"),
            (header + ",r1,0,1\n", "identifiers"),
            (header + " a,r1,0,1\n", "identifiers"),
            (header + '"a\nb",r1,0,1\n', "control characters"),
            (header + '"a,r1,0,1\n', "Malformed CSV"),
            (header, "measurement"),
            ("specimen_id,replicate_id,label,label,feature_x\na,r1,0,0,1\n", "unique"),
            ("specimen_id,replicate_id,label,batch_id,feature_x\na,r1,0,b1,1\n", "Unsupported"),
            ("specimen_id,replicate_id,label\na,r1,0\n", "feature"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.csv"
            for content, message in cases:
                with self.subTest(message=message, content=content):
                    path.write_text(content)
                    with self.assertRaisesRegex(ValueError, message):
                        read_specimen_csv(path)

    def test_prediction_optional_labels_but_not_partial_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "new.csv"
            path.write_text("specimen_id,replicate_id,feature_a\na,r1,2\n")
            self.assertIsNone(read_specimen_csv(path, require_labels=False).y)
            with self.assertRaisesRegex(ValueError, "label"):
                read_specimen_csv(path)
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                read_specimen_csv(path, require_labels=False, expected_features=("feature_b",))
            path.write_text("specimen_id,replicate_id,label,feature_a\na,r1,,2\n")
            with self.assertRaisesRegex(ValueError, "label"):
                read_specimen_csv(path, require_labels=False)

    def test_four_roles_are_disjoint_complete_and_reproducible(self):
        partitions = split_specimens(60, 4)
        self.assertEqual([len(rows) for rows in partitions.values()], [30, 9, 9, 12])
        np.testing.assert_array_equal(np.sort(np.concatenate(list(partitions.values()))), np.arange(60))
        for name, rows in partitions.items():
            np.testing.assert_array_equal(rows, split_specimens(60, 4)[name])
        with self.assertRaisesRegex(ValueError, "60"):
            split_specimens(59, 4)


class ToolkitTests(unittest.TestCase):
    def test_fit_restore_and_predict_new_specimens(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = generate_example(root / "source.csv", specimens=60, seed=7)
            metrics = fit_dataset(source, root / "model", epochs=4, seed=9, alpha=.2,
                                  hidden=(5, 4), head_width=3)
            self.assertEqual(metrics["n_test_specimens"], 12)
            model, scaler, metadata, threshold = load_model(root / "model")
            self.assertEqual(metadata["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            data = read_specimen_csv(source)
            train_ids = set(metadata["split_specimen_ids"]["train"])
            train = np.array([i for i, value in enumerate(data.specimen_ids) if value in train_ids])
            np.testing.assert_allclose(scaler.mean_, data.X[train].mean(axis=(0, 1)))
            calibration_ids = set(metadata["split_specimen_ids"]["calibration"])
            cal = np.array([i for i, value in enumerate(data.specimen_ids) if value in calibration_ids])
            scaled = scaler.transform(data.X, data.mask)
            expected = conformal_threshold(model.predict_proba(scaled[cal], data.mask[cal]), data.y[cal], .2)
            self.assertEqual(threshold, expected)
            with (root / "model" / "holdout_predictions.csv").open() as handle:
                holdout = list(csv.DictReader(handle))
            row_order = [data.specimen_ids.index(row["specimen_id"]) for row in holdout]
            expected = model.predict_proba(scaled[row_order], data.mask[row_order])
            np.testing.assert_array_equal([float(row["p_label1"]) for row in holdout], expected)
            fresh = generate_example(root / "fresh.csv", specimens=6, seed=8)
            result = predict_dataset(root / "model", fresh, root / "predictions.csv")
            self.assertEqual(result["n_specimens"], 6)
            with self.assertRaises(FileExistsError):
                fit_dataset(source, root / "model", epochs=1)
            with self.assertRaises(FileExistsError):
                predict_dataset(root / "model", fresh, root / "predictions.csv")
            with self.assertRaises(FileExistsError):
                generate_example(source)
            with self.assertRaisesRegex(ValueError, "previously used"):
                predict_dataset(root / "model", source, root / "overlap.csv")
            self.assertFalse((root / "overlap.csv").exists())

    def test_feature_reordering_preserves_predictions_and_renaming_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = generate_example(root / "train.csv", specimens=60, seed=2)
            fit_dataset(source, root / "model", epochs=2, hidden=(3, 3), head_width=2)
            fresh = generate_example(root / "fresh.csv", specimens=2, seed=3)
            with fresh.open() as handle:
                reader = csv.DictReader(handle)
                columns, rows = reader.fieldnames, list(reader)
            reordered = root / "reordered.csv"
            with reordered.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(reversed(columns)))
                writer.writeheader()
                writer.writerows(rows)
            predict_dataset(root / "model", fresh, root / "p1.csv")
            predict_dataset(root / "model", reordered, root / "p2.csv")
            self.assertEqual((root / "p1.csv").read_bytes(), (root / "p2.csv").read_bytes())
            reordered.write_text(reordered.read_text().replace("feature_0", "feature_new"))
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                predict_dataset(root / "model", reordered, root / "p3.csv")

    def test_calibration_and_test_labels_cannot_change_fitted_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = generate_example(root / "source.csv", specimens=60, seed=42)
            options = dict(epochs=4, hidden=(4, 3), head_width=2, seed=15)
            fit_dataset(source, root / "model1", **options)
            model1, scaler1, metadata, _ = load_model(root / "model1")
            protected = set(metadata["split_specimen_ids"]["calibration"] + metadata["split_specimen_ids"]["test"])
            with source.open() as handle:
                reader = csv.DictReader(handle)
                columns, rows = reader.fieldnames, list(reader)
            for row in rows:
                if row["specimen_id"] in protected:
                    row["label"] = str(1 - int(row["label"]))
            mutated = root / "changed_labels.csv"
            with mutated.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            fit_dataset(mutated, root / "model2", **options)
            model2, scaler2, _, _ = load_model(root / "model2")
            np.testing.assert_array_equal(scaler1.mean_, scaler2.mean_)
            for name in model1.parameters_:
                np.testing.assert_array_equal(model1.parameters_[name], model2.parameters_[name])
            self.assertEqual(model1.best_epoch_, model2.best_epoch_)

    def test_model_payload_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = generate_example(root / "source.csv", specimens=60)
            fit_dataset(source, root / "model", epochs=1, hidden=(2, 2), head_width=2)
            payload = root / "model" / "weights.npz"
            payload.write_bytes(payload.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_model(root / "model")

    def test_invalid_model_metadata_is_rejected_and_infinite_sentinel_roundtrips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = generate_example(root / "source.csv", specimens=60)
            fit_dataset(source, root / "model", epochs=1, alpha=.01, hidden=(2, 2), head_width=2)
            _, _, metadata, threshold = load_model(root / "model")
            self.assertTrue(np.isposinf(threshold))
            self.assertEqual(metadata["conformal_threshold"], "+inf")
            self.assertIn("python_version", metadata["software"])
            self.assertIn("replicateshift_version", metadata["software"])
            invalid = [[], dict(metadata, feature_names=[["feature_0"], "feature_1", "feature_2"]),
                       dict(metadata, conformal_threshold=True), dict(metadata, conformal_threshold=float("inf")),
                       dict(metadata, alpha=True),
                       dict(metadata, split_specimen_ids={"train": "abc", "selection": ["d"],
                                                         "calibration": ["e"], "test": ["f"]})]
            for entry in invalid:
                with self.subTest(entry=entry):
                    (root / "model" / "model.json").write_text(json.dumps(entry))
                    with self.assertRaises(ValueError):
                        load_model(root / "model")


if __name__ == "__main__":
    unittest.main()
