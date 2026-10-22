# Validation record — v0.2, 2026-10-22

These checks were actually executed against the implemented toolkit and allocation study. They establish the reported software behavior and simulation results, not real-data validity or methodological novelty.

## Environment and suite

Python 3.12.14, NumPy 2.3.5, macOS ARM64, using an existing runtime. No local software was installed. Public examples and experiments use synthetic data only.

`PYTHONPATH=src python -m unittest discover -s tests -v`: **62 tests passed** in an independent review execution (approximately 0.40 seconds).

The suite includes every-parameter finite-difference checks, repeat-order/padding invariance, equal-specimen scaling, deterministic training, minimum-validation-BCE checkpoint replay, conformal order statistics/ties/infinite thresholds, dataset/split integrity, fixed-budget allocation pairing, hand-computed paired MCSE, schema/persistence/ID guards, and tiny end-to-end artifact reconstructions. Tests mutate calibration/holdout labels and verify that fitted neural weights remain unchanged.

## Executed allocation pilot

The frozen protocol produced 180 model-condition runs and 54,000 test predictions. The first complete run recorded **8.48 seconds** and **18.45 MiB peak traced allocations**. Runtime measures the simulation/fitting/prediction loop and excludes final report serialization. Tracemalloc is not total resident or device memory. This is a small workload measurement, not a hardware performance benchmark.

An independent review verified:

- All nine source hashes and all seven output hashes in the manifest.
- Reconstruction of prediction-set membership, Brier, accuracy, coverage, set size, and empty/singleton rates for every one of the 180 runs from compressed raw predictions.
- All 18 scenario/model summaries and 12 paired Brier differences/MCSEs against independent formulas.
- Regeneration of all 40 role pools from their saved seeds and agreement with pool hashes, ordered IDs, and raw test labels.
- Exact development assay budgets, additional test costs, separate role IDs, prefix sampling, and finite-sample calibration ranks.

A second complete execution reproduced `protocol.json`, `summary.csv`, `paired_comparisons.csv`, `predictions.csv.gz`, `REPORT.md`, and `report.html` **byte-for-byte**. Every per-run scientific metric, source hash, and role-pool provenance record also matched exactly. Created-at timestamps and measured timing are expected to differ.

The [manifest](../results/allocation_v0.2/manifest.json), [protocol](../results/allocation_v0.2/protocol.json), [report](../results/allocation_v0.2/REPORT.md), and [review](V0_2_REVIEW.md) retain the evidence. Published results were not selected to show a neural-model advantage, and no post-result hyperparameter retuning occurred.

## Reusable example

The two included CSVs regenerate byte-for-byte from their stated synthetic seeds. Saved-model predictions replay exactly for both internal holdout and fresh-specimen examples. The independent review reconstructed their metrics and confirmed numeric-only NPZ weights. Inspected public artifacts contain synthetic IDs and no private paths or credentials.

The main README, methods, example commands, console entry points, and CI steps were reviewed against the actual command-line parser. Fitting uses a positional input argument; prediction uses `--input`. Model parameters and fitted scaling arrays reside in NPZ; JSON stores configuration, histories, checksums, splits, and calibration metadata.

## Report presentation

The HTML report is generated from the same summary records as the Markdown report and CSV. It has no external scripts or services. A browser preview of the local HTML was blocked because the browser tool disallows `file:` URLs; rendered layout and interactive filtering were therefore not visually verified in this environment. The Markdown/CSV evidence is independently reconstructed as described above.

## Historical evidence and limits

The unchanged [v0.1 validation](VALIDATION_V0_1.md) records the earlier 28-test suite, three-seed shift/leakage smoke study, and successful hosted Python 3.10/3.12 CI. Its source hashes refer to that historical revision, not the current code. Current hosted CI status is visible in the [workflow](https://github.com/Chencharl/replicateshift/actions/workflows/tests.yml).

This release does not establish real RNA/LC-MS utility, robustness under higher-level batch dependence, informative repeat counts, optimal allocation, publication-level Monte Carlo precision, PyTorch/MPS performance, or a new inferential theorem. Those limits do not alter the executed results.
