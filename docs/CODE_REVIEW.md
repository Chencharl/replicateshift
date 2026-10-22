# Code Review — ReplicateShift prototype

Date: 2026-10-22

Repository / branch / commit: local ReplicateShift development snapshot; no commit was available for this review. This is an assistant source review during development, not an external scientific audit or assessment of the author's independent proficiency.

Files actually inspected: `src/replicateshift/{benchmark,inference,models,simulation}.py`, `tests/test_{benchmark,inference,models,simulation}.py`, and `pyproject.toml`. Review structure follows CREA's `templates/CODE_REVIEW_TEMPLATE.md`.

Tests/benchmarks actually executed:

- `PYTHONPATH=src python -m unittest discover -s tests -v`: **28 tests passed** in the inspected snapshot, including the 60-specimen, three-epoch artifact round-trip test.
- A targeted RNG probe reproduced identical simulator latent values and logistic initialization when the same seed was reused. The benchmark now separates these streams and has a regression test.
- API probes initially reproduced invalid Wilson intervals for a negative `z` and acceptance of fractional success counts. Both now raise `ValueError`.
- A one-epoch isolated run with mocked before/after source hashes rejected a source mismatch and did not write a completed manifest. The final rerun of the complete suite again passed all 28 tests.

Constraints / unavailable evidence: this review did not run an extensive simulation, inspect private research repositories, install software, or evaluate a real biological cohort. Cross-platform numerical identity, model optimality, subgroup coverage, and real batch dependence remain unverified. The main smoke-study report is a separate artifact; the reviewer did not use its results to tune the models.

## Executive assessment

- **Observed in code:** independent synthetic specimens have repeated noisy measurements, a shared binary label, and independent nuisance fingerprints. The main experiment splits specimens, averages measurements within each partition, and fits preprocessing only on training specimens. It compares linear logistic regression, logistic regression with all pairwise interactions, and a two-hidden-layer neural network.
- **Observed in code:** split conformal calibration uses one score per specimen, a finite-sample order statistic, and an infinity sentinel when the calibration sample is too small. Coverage is reported with set size, singletons, and empty sets. The report distinguishes marginal conformal coverage from the conditional target of its Wilson intervals.
- **Primary scientific risk:** a reproducible demonstration could be mistaken for a new inference method, proof of superior neural prediction, or evidence of biological validity. The current report explicitly limits all three claims. Paired assay-shift scenarios do not represent independent cohorts or retain a distribution-free coverage guarantee.
- **Primary engineering risk:** cross-platform replay and behavior at larger workloads remain unverified. Source changes during a run are now detected, and the identified interval-validation issue is corrected.
- **Best learning opportunity:** explain why the specimen is the experimental unit, then implement a test that demonstrates how a valid marginal coverage guarantee can coexist with poor coverage conditional on a particular fitted calibration threshold.

## Scores (0–4, evidence required)

These are provisional scores of the inspected prototype, not scores of the user's ability.

| Dimension | Score | Evidence |
|---|---:|---|
| Scientific correctness | 3 | The simulator states its mechanism and excludes RNA chemistry claims; the nonlinear comparator includes all feature pairs without using labels. No external biological validation is available. |
| Statistical validity | 3 | Specimen disjointness, training-only scaling, finite-sample conformal ranks, paired bootstrap resampling, and conditional interval scope are explicit. A small smoke study cannot establish broad comparative or subgroup claims. |
| Software design | 3 | Simulation, models, inference, and orchestration are separate; model gradients can be evaluated independently. The identified interval-validation issue is corrected. |
| Tests and reproducibility | 3 | 28 passing tests cover analytic gradients, split leakage, shift pairing, conformal edge cases, artifact consistency, and hashes. A targeted source-mismatch probe also passed; cross-platform replay remains unverified. |
| Performance | 2 | Nearest-neighbor distance calculation is chunked; the small end-to-end test completes. No realistic scaling study or process-level memory validation was performed. |
| Security and data governance | 3 | Inspected paths generate synthetic arrays and write local output files; no research data are needed. This was not a security audit of hosting or external accounts. |

## Findings

### 1. Data and model initialization reused random draws — resolved before the smoke study

- Severity: **medium**, resolved.
- File:line: corrected by `src/replicateshift/benchmark.py:32` and its use at `src/replicateshift/benchmark.py:169`.
- Evidence and impact: in the initial version, the same seed initialized simulation and the logistic model. With three features, one repeat, and zero noise, the initial logistic weights divided by 0.01 exactly equaled the first specimen's latent values. If that specimen was held out, initialization depended on held-out covariates, weakening the stated independent-training construction. No resulting performance inflation was demonstrated.
- Minimal change made: spawn separate role seeds for simulation, splitting, each model, bootstrap, and leakage diagnostics; record them in provenance. Shift scenarios deliberately reuse only the simulation stream to preserve the paired experiment.
- Test that proves the fix: `tests/test_benchmark.py:14` checks deterministic, distinct role seeds and different simulation/model random draws. The full suite passed after the change.
- Concept the user should learn: reproducibility and independence are separate requirements. Reusing a seed can be intentional for paired scenarios while being inappropriate across data generation and model initialization.

### 2. Report text ignored nondefault coverage and seed counts — resolved

- Severity: **low**, resolved.
- File:line: `src/replicateshift/benchmark.py:114` and `src/replicateshift/benchmark.py:134`.
- Evidence and impact: initial report prose hardcoded 90% coverage and three seeds even though CLI options could change both. This could contradict the actual run.
- Minimal change made: derive these statements from the saved configuration.
- Test that proves the fix: `tests/test_benchmark.py:45` verifies an 80% target and a one-seed run in the generated report.
- Concept the user should learn: report prose is part of the reproducibility interface and must track the numerical configuration.

### 3. Wilson interval utility accepted invalid counts and a negative critical value — resolved

- Severity: **low**, resolved; the benchmark already supplied valid integer counts and the default positive critical value.
- File:line: `src/replicateshift/inference.py:58`.
- Evidence and impact: initially, `wilson_interval(5, 10, z=-1)` returned `(0.6507556723, 0.3492443277)`, a reversed interval. `wilson_interval(1.5, 10)` also returned an interval despite a binomial success count being integer-valued. Reusing the public utility with invalid inputs could silently produce a meaningless result.
- Minimal change made: require integer, non-Boolean `successes` and `n`, retain `0 <= successes <= n`, and require a finite positive `z`.
- Test that proves the fix: `tests/test_inference.py:48` verifies that invalid counts and negative/nonfinite `z` raise `ValueError`, while valid edge results are preserved. The reviewer also reran the two original failure probes and confirmed rejection.
- Concept the user should learn: statistical domain restrictions belong in executable API contracts, not only annotations.

### 4. Source hashes initially described only completion-time files — resolved

- Severity: **low**, resolved for the identified source-change-during-run case; more material for long jobs with concurrent edits.
- File:line: `src/replicateshift/benchmark.py:164` and `src/replicateshift/benchmark.py:219`.
- Evidence and impact: initially, source files were hashed only after fitting and report generation. Python may already have imported an earlier version if a file changes during the run; a completion-time hash alone cannot detect that case. Output hashes themselves are checked by the artifact test.
- Minimal change made: capture source hashes before computation and compare at completion, refusing a completed manifest if they differ. Final publication still needs a frozen source snapshot and a fresh verification run.
- Test that proves the fix: a reviewer-run, one-epoch integration probe mocked different before/after source hashes. It raised the intended `RuntimeError` and produced no `manifest.json`; the full 28-test suite still passed. No actual source file was modified by the probe.
- Concept the user should learn: provenance must bind the execution to a source snapshot, not simply record whichever files exist afterward.

## Learning assignment

- **One theory concept:** distinguish exchangeability of calibration/test specimens, marginal conformal coverage, and the conditional coverage probability estimated after fixing one calibration set.
- **One user-implemented coding exercise:** without changing model hyperparameters, add a small nested simulation that draws several independent calibration sets for a fixed predictor and evaluates each threshold on a separate large synthetic test sample. Predict the distribution of conditional coverage before running it.
- **Acceptance criteria:** state the random variables over which each result averages; preserve specimen independence; report between-calibration variation separately from finite-test sampling error; explain why individual calibration sets need not each achieve exactly 90% coverage.
- **Hints before full solution:** begin with a fixed analytic score function to remove training variability. Draw new calibration data for each repetition, keep one coverage estimate per calibration set, and do not treat all test outcomes sharing a threshold as independent evidence about marginal variability.

## Next actions

1. Finish the release owner's verification against the frozen source and preserve the final manifest. All four findings above have been corrected and checked in the reviewer-inspected snapshot.
2. Use the existing smoke run to check the workflow and formulate a preregistered larger simulation; do not select hyperparameters from its test results.
3. Have the user explain and implement one methodological extension before describing the project as evidence of independently mastered methods.
