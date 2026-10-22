# Code review — v0.2 independent project delivery

Date: 2026-10-22. Scope: the v0.2 working tree based on commit `4404f34864cd9bce449dcdb4a0c42a340a64d831`.

This review evaluates implementation and the synthetic experimental design. It does not establish methodological novelty, real-data validity, or the project owner's independent mastery. The review was performed with AI assistance separately from the implementation tasks.

## Inspected evidence

Source files actually inspected: `set_model.py`, `dataset_io.py`, `toolkit.py`, `allocation.py`, and the validation-checkpoint changes in `models.py`. The new model, ingestion, and persistence tests were inspected. The fixed allocation configuration was also read directly; the existing inference, simulation, and benchmark tests were executed as part of the original suite.

Independent verification used Python 3.12.14 and NumPy 2.3.5. All 62 tests in the complete v0.2 suite passed in 0.396 seconds. Before the extension, all 28 original tests also passed. Focused reviewer checks verified both legacy model classes' exact checkpoint replay, unchanged training trajectories under conflicting validation labels, and reset on refitting. Allocation-helper checks verified nested specimens and repeats, paired noise scaling, zero SD at one repeat, role-prefixed identifiers, exactly 600 development assay rows in every prescribed allocation, and paired-difference MCSE against a hand-calculated two-seed fixture.

The completed `results/allocation_v0.2` artifacts were independently audited after execution:

- All nine source hashes and seven output hashes match the saved manifest.
- All 54,000 saved prediction rows reconstruct Brier score, accuracy, coverage, mean set size, empty-set rate, singleton rate, set membership, and assay costs for all 180 model-condition runs. These rows reuse paired test specimens across models and conditions; they are not 54,000 independent specimens.
- The 18 summary rows and 12 paired model comparisons reproduce from per-run metrics, including every reported Monte Carlo standard error.
- All 40 maximum role pools reproduce exactly from their saved role seeds. Test labels match the prediction rows, and role IDs are disjoint within each seed. A specimen's experiment-wide identity includes the seed.
- The recorded run took 8.476 seconds, with a peak of 18.453 MiB in traced allocations. That figure does not measure total resident memory.
- Both public example input CSVs regenerate byte for byte from the synthetic generator. The saved model exactly reproduces its 24-specimen holdout predictions and 30 new-specimen predictions; their main metrics reconstruct from those predictions. The NPZ contains numeric arrays only. Inspected metadata retains synthetic identifiers, input basenames, and software versions, with no private filesystem paths or credentials found.

The pilot's complete results and runtime belong to its `manifest.json` and `REPORT.md`; passing tests alone is not evidence of model superiority. Current full-suite counts and CI outcomes are recorded in `VALIDATION.md`.

## Assessment

**Observed:** the extension provides a trained, permutation-invariant model of technical-repeat sets, a local CSV fit/predict workflow with reusable model artifacts, and a reproducible fixed-budget experiment with two nonlinear statistical baselines. It is an independent software/research project with concrete outputs.

The independent prediction unit is a specimen. Training, checkpoint selection, conformal calibration, and final evaluation have distinct roles. The allocation simulator keeps those roles fixed across policies using independent maximum pools and nested prefixes. Both logistic baselines include the same mean-feature interactions; one additionally includes within-specimen SD, so that comparison does not confound retaining SD with removing the generating interaction.

| Dimension | Score / 4 | Evidence and limit |
|---|---:|---|
| Scientific / algorithmic correctness | 3 | Masked gradients agree with numerical derivatives; repeat-order and padding invariance are tested. Features are synthetic and do not implement RNA chemistry. |
| Statistical validity | 3 | Distinct specimen roles, training-only preprocessing, finite-sample conformal rank, and paired run-level comparisons are explicit. Ten seeds support an exploratory pilot. |
| Software design | 3 | Ingestion, preprocessing, model, inference, toolkit, and experiment are separate modules; model bundles have a versioned schema. Public APIs remain intentionally narrow. |
| Tests / reproducibility | 3 | Numerical, behavioral, persistence, leakage, and artifact tests accompany deterministic streams and source/data/output hashes. Cross-platform results still depend on the recorded numerical environment. |
| Performance | 2 | Full-batch NumPy is appropriate for the small prescribed experiment; an explicit runtime cap limits the run. No large-data throughput or resident-memory claim is supported. |
| Security / data governance | 3 | Local processing, non-pickle array loading, schema/checksum validation, and refusal to overwrite outputs are implemented. Plain specimen IDs are retained for overlap checks and provenance. |

Scores assess this project's stated scope, not research novelty or biological readiness.

## Findings and disposition

No blocker was found in the inspected specimen-level inference or masked-gradient implementation. The following limitations remain explicit design boundaries.

1. **Medium — allocation interpretation.** `allocation.py` changes repeat count for both development and external testing. The test cost is `300 × repeats` additional assays. The results therefore compare complete measurement policies, not a training-only intervention. Calibration sample sizes also fall from 120 to 40 to 20, changing conformal rank resolution. The report must retain these costs and sample counts; the budget counts equal-cost assay rows and does not price specimen acquisition or preparation.

2. **Medium — generalization beyond the simulator.** `allocation.py:64` defines conditionally independent Gaussian technical noise and a known latent interaction. Within-specimen SD is largely ancillary under this observation model. The interaction baseline is deliberately strong, and a Deep Sets win is not a release requirement. Validation on a different observation process or approved real data is needed before claiming practical benefit.

3. **Low — repeat-count information.** `set_model.py:205` uses a masked mean. Duplicating an entire repeat set leaves the model score unchanged, so repeat cardinality is not a separate predictive input. The existing duplication-invariance test verifies that contract. Fixed repeat count within each pilot arm is compatible with this choice; variable-count applications should not assume that the network automatically models the amount of evidence supplied by the repeat count.

4. **Medium — identity is not independence.** `toolkit.py` rejects prediction IDs seen in any development or holdout role, but renaming an ID can bypass that check, and distinct specimens may still share patients, families, or batch effects. The workflow requires the stated independent-specimen and exchangeability assumptions. It does not support a higher sampling hierarchy merely because a CSV can be parsed.

5. **Low — parser and artifact robustness, resolved during review.** Multiline CSV identifiers originally risked losing line breaks during parsing; ingestion now preserves CSV structure and rejects unsupported control characters. Saved metadata validation was tightened for its top-level type, feature names, partition lists, and conformal threshold representation. Relevant regression cases accompany these changes. The report now directs reproduction to the actual saved protocol, distinguishes per-run selected epochs from manifest contents, and describes the costs excluded from the assay budget.

For larger or heterogeneous datasets, the next scientific extension should be chosen from an observed failure mode. Current outputs support inspection and reproduction; they do not justify an optimal laboratory allocation rule or a new coverage theorem.
