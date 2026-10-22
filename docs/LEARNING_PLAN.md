# ReplicateShift learning plan

Created 2026-10-22. Proposed workload: 10–12 hours per week. This plan adapts the CREA `templates/WEEKLY_SESSION.md` contract; it does not replace the broader roadmap or record completed learning.

The repository's current status reports prior biostatistics training and Python research experience, but does not establish proficiency with neural-network gradients, calibration, leakage assessment, or simulation-study design. Use the actual attempt at the first exercise as the diagnostic. The existence of assistant-written code is not evidence that the learner can explain or reproduce it.

## First task: recognize the independent unit

**Effort:** 45–60 minutes. **Prerequisites:** Python arrays and sets, training/test splits, and the distinction between a specimen and a repeated measurement. No new package or real laboratory data is needed.

**Concept:** technical repeats share specimen information. A row split can allow a model to recognize a specimen already represented in training. An evaluation for a new specimen requires splitting whole specimens.

Before inspecting the implementation or running the experiment, write a prediction in your own words:

1. With three nearly identical measurements per specimen, will a random row split usually put the same specimen into both training and test?
2. If labels are randomly reassigned between specimens and then copied to each specimen's repeats, could nearest neighbor still predict held-out rows well? What changes for a whole-specimen test split?
3. What does “n” count in a confidence interval for prediction on new specimens?
4. If two specimens share a donor or random acquisition-batch effect, is a specimen split necessarily sufficient?

Then independently implement a small function that accepts a vector of specimen IDs and returns disjoint training and test specimen IDs plus the corresponding row indices. Sketch its invariants first. Use a tiny synthetic fixture containing repeated IDs; do not start by copying the project splitter.

**Acceptance tests:**

- Every row is assigned exactly once to the intended partition.
- The intersection of training and test specimen IDs is empty.
- All repeats of one specimen stay together.
- A fixed seed gives the same split; changing row order does not invalidate the disjointness invariant.
- You can explain why a row-wise label permutation is a different negative control from a specimen-wise permutation.

**Evidence to retain:** the prediction, your function, the checks actually run, and a short explanation of any mismatch between prediction and output. The mentor reviews that attempt, gives targeted hints, and provides a complete solution only if requested or if you remain blocked. The checks establish software behavior, not statistical coverage or biological validity.

## Weekly Learning Contract — week 1, beginning 2026-10-22

### Starting state

- Relevant roadmap topics: weeks 3, 20, 27, and 41, brought together in a bounded capstone exercise; this does not mark those weeks complete.
- Current repository: CREA_Codex; project subdirectory `projects/replicateshift`. Record the actual Git state when beginning the session; no branch is assumed here.
- Verified learning progress since last week: none recorded for this project.
- Time available: 10–12 hours, provisional.

### Goal and rationale

- One concrete deliverable: a learner-authored explanation and tested splitter that distinguish measurement-row leakage from new-specimen prediction, accompanied by an honest reading of the prototype output.
- RNA/LC-MS relevance: multiple injections or measurements do not create independent specimens; an evaluation must match the eventual scientific use.
- PhD/industry relevance: translate an estimand into a split strategy, encode invariants, diagnose misleading performance, and communicate a model's limits.

### Plan (12 hours; reduce reading/reflection by 2 hours if needed)

- Theory, 4h: first-task exercise; independent-unit reasoning; exchangeability; prediction sets versus confidence intervals; work through the conformal rank with a hand-calculated example.
- Independent implementation, 4h: write the specimen splitter and tests; trace training-only standardization; write pseudocode for the score and quantile without consulting the implementation first.
- Testing and experimental validation, 2h: run existing bounded smoke checks; inspect one split and one set construction; compare row-split and specimen-split negative controls without changing parameters to obtain a desired result.
- Reading/career/reflection, 2h: read the tutorial introduction and the hierarchical-data paper's problem formulation; draft a 150-word portfolio explanation stating the question, method, and limits.

### Acceptance tests

- Correctness: splitter invariants pass; reproduce a conformal quantile by hand, including ties and the k > n case.
- Statistical validity: explain why the coverage claim targets a new specimen; name the exchangeability condition; identify why marginal coverage is not subgroup or conditional coverage.
- Reproducibility: retain the exact configuration, seeds, command, interpreter version, and outputs from the checks actually run.
- Performance: no hardware or expensive benchmark. Record bounded runtime only if measured.

### End-of-week evidence

- Code changes (paths/commit): to be filled after the learner's work; no commit is assumed.
- Tests actually run: to be filled from the command log.
- Benchmark results: smoke results only, if run; larger study not started.
- What I can now explain without help: learner writes this after the review.
- Next unresolved question: does the observed gap arise from split leakage, representation choice, calibration randomness, or shift?

## Weekly Learning Contract — week 2

### Starting state

- Begin only after week 1 evidence is reviewed. Carry unresolved issues forward.
- Relevant roadmap topics: statistical observation models, neural-network foundations, and uncertainty evaluation.
- Time available: 10–12 hours.

### Goal and rationale

- One concrete deliverable: an independently checked gradient derivation for a small network and a paired predictive-performance analysis.
- RNA/LC-MS relevance: understand how a flexible score is learned before assigning scientific meaning to it.
- PhD/industry relevance: inspect numerical algorithms, distinguish optimization from inference, and compare models fairly.

### Plan (12 hours)

- Theory, 4h: binary cross-entropy; chain rule; two hidden layers; regularization; Brier score; conditional versus procedure-wide uncertainty.
- Independent implementation, 4h: derive one layer's forward and backward calculation and implement a finite-difference check on a tiny fixture; trace the existing model only after making a prediction.
- Testing and experimental validation, 2h: compare numerical and analytic gradients; verify paired bootstrap indices keep each model pair's predictions for a specimen together; inspect linear and nonlinear toy examples.
- Reading/career/reflection, 2h: explain in a short technical note why a neural network may lose to logistic regression with interaction features, why a linear-only comparator can be misleading, and why an uncertain prediction set is not a failed implementation.

### Acceptance tests

- Correctness: gradient checks pass a declared numeric tolerance on a nondegenerate fixture; probabilities are finite and within [0, 1].
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

After the first three contracts, choose one extension rather than expanding every direction at once:

- **Measurement allocation:** compare more independent specimens against more technical repeats at fixed total cost, with a scientifically defined utility and adequate complete-run precision.
- **Representation:** compare mean-plus-variance/count with a permutation-invariant neural encoder after library and installation approval; preserve identical independent splits and calibration rules.
- **Inference:** align a published hierarchical or shift-aware method with the exact target before implementing it; specify any changed assumption and independently review a new proof.

An approved external dataset and a wider literature review are prerequisites for claims about transferability or novelty. Do not use laboratory data merely because it is available elsewhere on the machine.

## Progress update proposal

At the end of each substantive session, propose an addition to `progress/STATUS.md` with the date, completed work, concrete file/test/output evidence, skills practiced, unresolved issues, and the next small task. Only list work that occurred. In particular:

- “A prototype and research plan exist” is software-delivery evidence.
- “I implemented the split and explained its independent-unit assumption without help” requires the learner's attempt and review.
- “The MLP improves inference” requires an aligned estimand, sufficient evaluation, and a more precise statement of what improved.

No learning completion or scientific finding is recorded by this document itself.
