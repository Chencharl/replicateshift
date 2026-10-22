# ReplicateShift: synthetic smoke study

Status: executed prototype validation. This is not a powered simulation study or biological validation.

Seeds: [7, 19, 43]; 600 specimens per seed, 3 technical repeats each. Train/calibration/test: 60/20/20% by specimen. Target coverage: 90%.

## Specimen-level results

Each entry is the arithmetic mean across seeds. Read per_seed_metrics.csv for every run, coverage intervals, and uncertainty diagnostics.

| Model | Assay shift | Accuracy | Brier ↓ | Coverage | Mean set size | Singleton rate |
|---|---:|---:|---:|---:|---:|---:|
| interaction_logistic | 0 | 0.750 | 0.190 | 0.908 | 1.494 | 0.506 |
| interaction_logistic | 1.5 | 0.597 | 0.299 | 0.833 | 1.494 | 0.506 |
| logistic | 0 | 0.683 | 0.207 | 0.928 | 1.644 | 0.356 |
| logistic | 1.5 | 0.617 | 0.229 | 0.847 | 1.469 | 0.531 |
| mlp | 0 | 0.653 | 0.297 | 0.900 | 1.592 | 0.408 |
| mlp | 1.5 | 0.636 | 0.331 | 0.847 | 1.597 | 0.403 |

Coverage must be read with set size: predicting both labels always covers the true label. The 90% guarantee is marginal over exchangeable calibration and new test specimens, conditional on independent training. It is not a per-specimen, subgroup, fixed-calibration, or shifted-distribution guarantee.

## Leakage negative control

Labels were permuted across specimens and repeated within each specimen. A 1-nearest-neighbor model tests whether specimen fingerprints can cross the split. These row-level accuracies describe different prediction tasks; they are not comparable to the table above.

| Seed | Split | Accuracy per row | Test rows from training specimens |
|---:|---|---:|---:|
| 7 | row_split | 0.900 | 0.825 |
| 7 | specimen_split | 0.475 | 0.000 |
| 19 | row_split | 0.903 | 0.828 |
| 19 | specimen_split | 0.508 | 0.000 |
| 43 | row_split | 0.911 | 0.836 |
| 43 | specimen_split | 0.506 | 0.000 |

## Paired model comparison

Difference in test Brier score: MLP minus each baseline. Negative favors the MLP. 95% percentile bootstrap intervals resample test specimens jointly for both models, conditional on the fitted models. They do not include retraining uncertainty or establish significance across the full benchmark.

| Seed | Assay shift | Baseline | Brier difference | Conditional interval |
|---:|---:|---|---:|---|
| 7 | 0 | logistic | 0.032 | [-0.026, 0.091] |
| 7 | 0 | interaction_logistic | 0.064 | [0.013, 0.118] |
| 7 | 1.5 | logistic | 0.058 | [-0.015, 0.130] |
| 7 | 1.5 | interaction_logistic | -0.016 | [-0.083, 0.048] |
| 19 | 0 | logistic | 0.073 | [0.005, 0.135] |
| 19 | 0 | interaction_logistic | 0.094 | [0.018, 0.165] |
| 19 | 1.5 | logistic | 0.071 | [0.017, 0.127] |
| 19 | 1.5 | interaction_logistic | 0.075 | [0.011, 0.147] |
| 43 | 0 | logistic | 0.166 | [0.089, 0.243] |
| 43 | 0 | interaction_logistic | 0.163 | [0.080, 0.250] |
| 43 | 1.5 | logistic | 0.178 | [0.102, 0.256] |
| 43 | 1.5 | interaction_logistic | 0.037 | [-0.054, 0.124] |

## Interpretation limits

- The neural network, linear model, and conformal procedure are established methods. The project contribution is the reproducible experiment and its research extension.
- Only synthetic features and labels were used. The simulator does not model RNA chemistry, actual mass spectra, or clinical outcomes.
- The shift changes measured features with the latent specimen labels held fixed. It need not preserve P(Y|X).
- Shift scenarios reuse paired latent test specimens. Do not treat scenarios as independent replicates.
- This run uses 3 seeds. Summary MCSE values are descriptive; the default three-seed run is a smoke study, and no superiority claim is made.
- Wilson intervals describe independent test outcomes conditional on each fixed fit and calibration set; calibration variability is assessed only through repeated seeds.
- The simulator has independent specimens and no shared random batch effect. Real batch dependence requires additional splitting and inferential design.
- Runtime is measured for this smoke run. tracemalloc peaks are traced allocations, not total resident or device memory.

## Reproduce

```sh
PYTHONPATH=src python -m replicateshift.benchmark --specimens 600 --replicates 3 --epochs 180 --alpha 0.1 --bootstrap 1000 --seeds 7 19 43 --shifts 0.0 1.5 --output results/local-reproduction
```

Use a new output directory; the CLI refuses to overwrite existing results. manifest.json records configuration, source hashes, data/split hashes, software versions, time, and memory scope.
