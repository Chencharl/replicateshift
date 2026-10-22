# ReplicateShift learning plan

Created 2026-10-22. Proposed workload: 10–12 hours per week. This plan adapts the CREA `templates/WEEKLY_SESSION.md` contract; it does not replace the broader roadmap or record completed learning.

Updated 2026-10-22 after the initial diagnostic and a read-only review of public projects. The starting point is applied Python/statistical modeling with experience adapting pretrained models. Public source includes LoRA fine-tuning through Hugging Face Trainer, gradient-based robustness experiments, and RNA constraint/QC implementations. Fine-tuning is model-training experience; it does not by itself establish training-from-scratch or independent mastery of every component in a collaborative project.

The diagnostic correctly identified that technical repeats from one original specimen belong in the same split, and that marginal 90% coverage is neither 90% accuracy nor a guarantee for every subgroup or of narrow prediction sets. Introductory split mechanics are therefore a brief check, not the first week's main exercise. The learning priority is experimental design: model selection, calibration, and final evaluation under a declared measurement protocol.

## First task: design the model-selection and calibration contract

**Effort:** 45–60 minutes. **Prerequisites:** existing fine-tuning experience, loss/optimizer concepts, specimen grouping, and marginal prediction coverage. No package installation or real laboratory data is required. This is question 1 of a three-question applied sequence; answer and review one question at a time.

**Concept:** a set used to compare learning rates or select epochs serves model development, regardless of whether it is named `test`. Conformal calibration introduces another role with its own restrictions. The already-inspected v0.1 test results are exploratory evidence for future design, not an untouched test for a redesigned model.

**Predict before implementation:** for a repeat-aware neural model, sketch where you would fit weights, choose hyperparameters/early stopping, estimate a conformal threshold, and report final performance. Which specimen IDs and labels may each decision access? Explain how a shared random batch effect would change the independent unit.

Use the current source as the diagnostic. Draw the proposed split/decision table and list its invariants before writing code. After the mentor reviews the attempt, implement a small split-audit helper on synthetic IDs. Keep the existing v0.1 results unchanged.

**Acceptance tests:**

- Every specimen belongs to exactly one declared partition; all of its repeats follow it.
- The design explicitly distinguishes fitting, selection, calibration, and final evaluation; any reuse is justified by an applicable procedure.
- No final-test result chooses an epoch, representation, preprocessing rule, or score.
- The calibration score and fitted prediction rule are frozen before calibration labels are used.
- Metrics weight the intended independent units, and uncertainty denominators do not count technical repeats as additional specimens.
- The next experiment reserves a fresh evaluation stream after the protocol is fixed; it does not relabel the published v0.1 results as untouched evaluation.

**Evidence to retain:** the initial design, reviewed revision, audit helper, and checks actually run. The mentor gives targeted hints before a full solution. Questions 2 and 3, addressed later, concern changing repeat-collection policies and measuring uncertainty across independent training/calibration repetitions.

## Weekly Learning Contract — week 1, beginning 2026-10-22

### Starting state

- Relevant roadmap topics: weeks 3, 20, 27, and 41, brought together in a bounded capstone exercise; this does not mark those weeks complete.
- Current repository: CREA_Codex; project subdirectory `projects/replicateshift`. Record the actual Git state when beginning the session; no branch is assumed here.
- Verified conceptual starting point: diagnostic answers correctly distinguish specimen grouping, marginal coverage, accuracy, subgroup coverage, and informativeness. Implementation mastery remains an applied assessment.
- Time available: 10–12 hours, provisional.

### Goal and rationale

- One concrete deliverable: a learner-authored evaluation contract and split-audit helper for selecting a neural model without contaminating calibration or final testing.
- RNA/LC-MS relevance: multiple injections or measurements do not create independent specimens; an evaluation must match the eventual scientific use.
- PhD/industry relevance: translate an estimand into model-development decisions, encode invariants, and defend a reproducible evaluation.

### Plan (12 hours; reduce reading/reflection by 2 hours if needed)

- Theory, 4h: selection versus evaluation; exchangeability conditional on a fixed independently trained rule; marginal versus realized-calibration coverage; a hand-calculated conformal rank example.
- Independent implementation, 4h: after review, write a split-audit helper and an explicit configuration for training-only model selection; trace which observations inform every fitted quantity.
- Testing and experimental validation, 2h: use tiny synthetic fixtures to catch specimen overlap, selection on test labels, and averaging unequal batch percentages; retain the frozen v0.1 outputs.
- Reading/career/reflection, 2h: align the new-specimen prediction target with conformal assumptions; draft a 150-word explanation of what fine-tuning experience transfers to this project and what experimental-design questions remain.

### Acceptance tests

- Correctness: split and decision-access invariants pass; reproduce a conformal quantile by hand, including ties and the k > n case; check an unequal-batch metric against total-correct/total-specimens.
- Statistical validity: justify the independent unit and where model selection occurs; explain what changes when calibration and test specimens receive different measurement protocols.
- Reproducibility: retain the exact configuration, seeds, command, interpreter version, and outputs from the checks actually run.
- Performance: no hardware or expensive benchmark. Record bounded runtime only if measured.

### End-of-week evidence

- Code changes (paths/commit): to be filled after the learner's work; no commit is assumed.
- Tests actually run: to be filled from the command log.
- Benchmark results: smoke results only, if run; larger study not started.
- What I can now explain without help: learner writes this after the review.
- Next unresolved question: under an identical collection policy, can a repeat-aware representation improve useful prediction-set size beyond simple specimen summaries?

## Weekly Learning Contract — week 2

### Starting state

- Begin only after week 1 evidence is reviewed. Carry unresolved issues forward.
- Relevant roadmap topics: statistical observation models, neural-network foundations, and uncertainty evaluation.
- Time available: 10–12 hours.

### Goal and rationale

- One concrete deliverable: an explicit small training-loop specification, a checked gradient calculation, and a paired predictive-performance analysis.
- RNA/LC-MS relevance: understand how a flexible score is learned before assigning scientific meaning to it.
- PhD/industry relevance: inspect numerical algorithms, distinguish optimization from inference, and compare models fairly.

### Plan (12 hours)

- Theory, 4h: connect Trainer-managed fine-tuning to optimizer steps, train/eval modes, parameter freezing, and checkpoint selection; review Brier score and conditional versus procedure-wide uncertainty.
- Independent implementation, 4h: trace one optimization step in the existing NumPy model and independently verify one layer's gradient. Specify a future PyTorch equivalent without installing dependencies or treating it as implemented.
- Testing and experimental validation, 2h: compare numerical and analytic gradients; verify paired bootstrap indices keep each model pair's predictions for a specimen together; inspect linear and nonlinear toy examples.
- Reading/career/reflection, 2h: explain in a short technical note why a neural network may lose to logistic regression with interaction features, why a linear-only comparator can be misleading, and why an uncertain prediction set is not a failed implementation.

### Acceptance tests

- Correctness: gradient checks pass a declared numeric tolerance on a nondegenerate fixture; distinguish input-gradient attacks from parameter training; probabilities are finite and within [0, 1].
- Statistical validity: identify which random quantities are held fixed by the test-set bootstrap and which would vary under complete dataset replication.
- Reproducibility: rerun the same small configuration; explain any allowed floating-point tolerance.
- Performance: discuss the model's approximate compute and memory scaling without claiming an unmeasured speedup.

### End-of-week evidence

- Retain derivation, code diff, numerical checks, observed model comparison, and a learner-written explanation. Record failures as failures.
- Next unresolved question: which observed model difference is stable across independently generated datasets?

## Weekly Learning Contract — week 3

### Starting state

- Begin after the model and split checks are understood. Larger benchmark approval and runtime budget remain prerequisites.
- Relevant roadmap topics: simulation design, calibration, statistical reporting, and reproducibility.
- Time available: 10–12 hours.

### Goal and rationale

- One concrete deliverable: a preregistered simulation specification with a precision calculation and a small, explicitly exploratory pilot if approved.
- RNA/LC-MS relevance: plan how many independent specimens and repeated measurements are needed to answer a stated question.
- PhD/industry relevance: design an informative experiment before consuming compute or interpreting a plot.

### Plan (12 hours)

- Theory, 4h: Monte Carlo standard errors; paired differences; precision versus power; model tuning and test reuse; shared batch dependence.
- Independent implementation, 4h: write an experiment specification and independent-run aggregation code; keep raw seed-level records and distinguish model randomness from simulator randomness.
- Testing and experimental validation, 2h: verify aggregation using hand-computed fixtures; if approved, run a small variance/runtime pilot and determine the later replication count.
- Reading/career/reflection, 2h: compare the target with existing hierarchical conformal work; write a novelty table listing what is known, implemented, proposed, and unverified.

### Acceptance tests

- Correctness: run-level MCSE calculation matches a hand-computed fixture; paired differences use identical generated datasets for competing methods.
- Statistical validity: define the unit of simulation replication, primary contrast, meaningful effect or precision target, and treatment of multiple comparisons before final results are examined.
- Reproducibility: a fresh run reconstructs all summaries from raw records and a saved configuration.
- Performance: runtime and memory budget are measured or explicitly unverified; obtain approval before an expensive run.

### End-of-week evidence

- Retain the frozen specification, planning calculations, actual pilot evidence if any, and a decision to continue, revise, or stop the proposed question.
- Next unresolved question: is there a useful contribution beyond demonstrating known behavior?

## Later work, gated by evidence

The selected domain is RNA/omics uncertainty and reproducibility. After the first three contracts, prioritize the measurement-allocation question, using representation and inference as components of that study:

- **Measurement allocation:** compare more independent specimens against more technical repeats at fixed total cost, with a scientifically defined utility and adequate complete-run precision.
- **Representation:** compare mean-plus-variance/count with a permutation-invariant neural encoder after library and installation approval; preserve identical independent splits and calibration rules.
- **Inference:** align a published hierarchical or shift-aware method with the exact target before implementing it; specify any changed assumption and independently review a new proof.

The proposed next benchmark compares mean-only, mean-plus-variance/count, and a permutation-invariant set encoder under prespecified specimen/repeat budgets. A [Deep Sets](https://arxiv.org/abs/1703.06114) architecture is an existing method, so the contribution must come from the measurement-design question and its evidence. First state the observation/collection protocol; a more expressive model does not repair an exchangeability mismatch. The inferential basis remains [split conformal prediction](https://arxiv.org/abs/2107.07511).

An approved external dataset and a wider literature review are prerequisites for claims about transferability or novelty. Do not use laboratory data merely because it is available elsewhere on the machine.

## Progress update proposal

At the end of each substantive session, propose an addition to `progress/STATUS.md` with the date, completed work, concrete file/test/output evidence, skills practiced, unresolved issues, and the next small task. Only list work that occurred. In particular:

- “A prototype and research plan exist” is software-delivery evidence.
- “I implemented the split and explained its independent-unit assumption without help” requires the learner's attempt and review.
- “The MLP improves inference” requires an aligned estimand, sufficient evaluation, and a more precise statement of what improved.

No learning completion or scientific finding is recorded by this document itself.
