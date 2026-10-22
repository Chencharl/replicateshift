# Validation record — 2026-10-22

This record reports checks actually executed for the initial release. It is evidence of software behavior, not evidence that the learner independently wrote or understands every component.

## Environment

- Python 3.12.14; NumPy 2.3.5; existing macOS ARM64 runtime.
- No local software was installed for this work.
- Synthetic data only; no external research repository or laboratory dataset was used.

## Executed checks

`PYTHONPATH=src python -m unittest discover -s tests -v`: **28 tests passed**.

The tests cover split disjointness and completeness, repeated-label integrity, deterministic generation, paired shift behavior, training-only scaling, analytic neural and logistic gradients against central finite differences, stable predictions, nonlinear held-out learning, conformal ranks/ties/small-calibration behavior, paired bootstrap identity, Wilson validation, separated random streams, all-pairs feature mapping, and CLI artifact reconstruction.

The default smoke benchmark was executed with 600 specimens, three repeats, 180 epochs, three seeds, two shift scenarios, three models, and 1,000 paired bootstrap resamples. Each seed used 360 training, 120 calibration, and 120 test specimens. See `results/smoke_v0.1/protocol.json` for the executed settings.

The final run recorded approximately **0.43 seconds** and **29.83 MiB peak traced allocations** in that environment. This is a bounded smoke workload, not a hardware benchmark. Traced allocations are not total resident memory.

A separate execution into a fresh directory reproduced the following files byte-for-byte: predictions, leakage results, paired Brier comparisons, summary metrics, report, and protocol. All per-seed scientific metrics also matched exactly after excluding fit timing. Manifest timestamps and timing fields are expected to differ.

SHA-256 readback checks verified every source and output hash recorded in the published manifest against the final local files. The package configuration parsed successfully. Local Markdown links were checked before publication.

## Review corrections

- Separated simulator and model-initialization streams using SeedSequence children; the initial shared-stream draft was excluded from release.
- Made alpha and seed-count wording reflect CLI inputs.
- Added all-pairwise-interaction logistic regression as a stronger simple nonlinear comparator before the reported study.
- Validated Wilson interval counts and positive finite z values.
- Recorded source hashes before fitting and rejected a run if source files changed during execution.

See `CODE_REVIEW.md` for the structured review. The final results retained the neural model's worse unshifted Brier score; parameters were not retuned after examining the test results.

## What remains unverified

- A powered, prespecified simulation study and broad seed stability.
- Real-data utility, clinical applicability, RNA/LC-MS fidelity, or biological findings.
- Independent learner mastery, methodological novelty, and any new statistical theorem.
- PyTorch, MPS/MLX, learned set encoders, higher-level random batches, informative repeat counts, and causal inference.

The initial commit `927927930599b1c9d2d5795bd132fbdd8e66bc88` also passed [GitHub Actions run 39798214597](https://github.com/Chencharl/replicateshift/actions/runs/39798214597): package installation, the 28-test suite, and a 120-specimen CLI smoke run succeeded on Python 3.10 and 3.12. This verifies those workflows on the hosted Linux environment; it does not establish bitwise numerical identity with the macOS results.
