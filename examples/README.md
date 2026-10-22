# Executed synthetic CSV example

These files demonstrate the actual `fit` → saved model → `predict` workflow. All identifiers, features, and labels are generated; no laboratory or patient data are included.

- `synthetic_assay.csv`: 120 specimens, three technical repeats, three numeric features, generator seed 2026.
- `new_specimens.csv`: 30 new specimens generated independently with seed 2027, three repeats each.
- `fitted_model/`: actual saved Deep Sets/scaler state, frozen conformal threshold, role IDs, software versions, input/weight hashes, and untouched holdout results.
- `predictions.csv`: probabilities and conformal label-set membership for those 30 new specimens.
- `execution.json`: recorded fit/prediction evaluation outputs.

Generate and fit from the repository root, using fresh paths:

```sh
replicateshift generate-example --output results/local-input.csv --specimens 120 --seed 2026
replicateshift generate-example --output results/local-new.csv --specimens 30 --seed 2027
replicateshift fit results/local-input.csv --output results/local-example-model
replicateshift predict --model results/local-example-model \
  --input results/local-new.csv --output results/local-example-predictions.csv
```

Reuse the included frozen model without retraining:

```sh
replicateshift predict --model examples/fitted_model \
  --input examples/new_specimens.csv --output results/local-reused-predictions.csv
```

If the package is not installed but NumPy is available, substitute `PYTHONPATH=src python -m replicateshift.toolkit` for `replicateshift`.

The 24-specimen holdout had Brier 0.103 and coverage 23/24; the 30 fresh specimens had Brier 0.229 and coverage 28/30. Mean label-set sizes were 1.458 and 1.767 respectively. These small samples demonstrate functioning artifacts and sample variability, not an estimate of robust real-world performance. This example has three features, whereas the allocation experiment has twelve, so its scores should not be compared with that study as competing methods.
