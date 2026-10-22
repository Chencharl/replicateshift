# Changelog

## 0.2.0 — 2026-10-22

ReplicateShift now provides a reusable CSV model toolkit and an executed measurement-allocation study.

- Added a masked Deep Sets encoder with specimen-weighted preprocessing, numerical gradient checks, validation checkpoint selection, and strict model-state restoration.
- Added `fit`, `predict`, and `generate-example` commands, variable-repeat CSV input, frozen conformal calibration, saved JSON/NPZ models, and specimen-overlap checks.
- Executed a fixed 600-assay development-budget pilot: ten seeds, two noise levels, three allocations, and three models. Published all 180 runs and 54,000 predictions, paired comparisons, source/output hashes, and a self-contained results table.
- Expanded verification from 28 to 62 tests; independently reconstructed metrics and repeated the complete experiment.
- Added an actual fitted-model example and current methods/validation documentation. Preserved v0.1 results and historical methods.

CLI change: `replicateshift` now opens the CSV toolkit. The original benchmark remains available as `replicateshift-benchmark` or `python -m replicateshift.benchmark`. The allocation runner is `replicateshift-allocation`.

The pilot does not show consistent neural-model superiority. This is synthetic research software, not a validated biological model or a new statistical theorem.

## 0.1.0 — 2026-10-22

Initial synthetic repeated-assay benchmark, specimen-level conformal inference, neural/logistic baselines, measurement-shift stress test, and permuted-label leakage control.
