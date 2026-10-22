# ReplicateShift research plan

Version: 0.1 design, 2026-10-22. This document separates the implemented prototype's intended scope from proposed research. It contains no experimental result or claim of novelty.

## Research question and prediction target

When repeated measurements come from the same specimen, how do the choice of independent sampling unit, nonlinear representation, and distribution shift affect the reliability and usefulness of predictive uncertainty?

The v0.1 target is the binary label of **one new independent specimen**, given its three technical measurements. A technical measurement is not an independent biological replicate. Each specimen contributes one mean feature vector, one label, and one calibration score. Every reported specimen-level denominator counts specimens, not measurement rows.

The initial deliverable is a small, inspectable research-software prototype and a reproducible stress test. A personal research identity can come from careful experimental design, clear failures, useful software, and an eventual focused contribution. Combining an MLP with split conformal prediction is not itself a new statistical method.

## What the first release can establish

| Status | Scope |
|---|---|
| Proposed v0.1 implementation | Synthetic data with 600 specimens, 3 technical repeats, 12 features; specimen splits of 60% training, 20% calibration, 20% test; seeds 7, 19, and 43. |
| Proposed model comparison | Linear logistic regression, logistic regression on 12 main effects plus all 66 pairwise products, and a NumPy MLP with two hidden layers. All use the same specimen means and splits, with training-only scaling of each model's input features. |
| Proposed predictive inference | Binary label sets from split conformal calibration at nominal coverage 90%. |
| Proposed stress experiment | Fixed test-feature shift strengths 0 and 1.5, using the same held-out latent specimens within each seed. |
| Proposed leakage control | Nearest-neighbor prediction after specimen-level label permutation, comparing random measurement-row splitting with whole-specimen splitting. |
| Unverified until recorded outputs exist | Numerical performance, neural-network advantage, leakage magnitude, test coverage, runtime, and reproducibility. |
| Outside v0.1 | Actual RNA identification, biomarker discovery, clinical utility, causal effects, parameter confidence intervals, new conformal theory, adaptive measurement collection, and real-data validation. |

The associated README and run artifacts must record which of these components actually passed their checks. This plan alone is not implementation evidence. Synthetic “omics-inspired” features are not a validated model of an LC-MS instrument or RNA biology.

## Sampling assumptions and inferential boundaries

For specimen i, write the repeated measurements as X_ij, j = 1,...,m_i, and the label as Y_i. The feature supplied to the v0.1 classifier is the within-specimen arithmetic mean. The complete objects (Y_i, m_i, X_i1,...,X_im_i) are independent and identically distributed in the unshifted simulation. Dependence within each object is permitted.

1. Split complete specimens before model fitting and calibration. The training, calibration, and test specimen-ID sets must be pairwise disjoint.
2. Learn scaling, model parameters, and any representation exclusively from training data. Fixed hyperparameters are preferable in this small prototype. Future tuning needs an inner validation split within training; calibration labels must not select a model or an epoch.
3. The calibration and new test specimen must be exchangeable under the deployment protocol for the usual split conformal guarantee. Apply the same feature definition and measurement protocol at both stages.
4. A different number or mechanism of technical repeats at deployment can change the distribution of the specimen summary. Future designs must state whether repeat count is fixed, random, informative about the specimen, or chosen adaptively.
5. If different specimens share a random donor, site, family, acquisition batch, or other supercluster effect, specimen IDs alone do not remove that dependence. Use the highest relevant independent unit in splitting and uncertainty estimation, or justify a method for the actual hierarchy. Aggregating technical repeats cannot resolve shared batch dependence.

A fixed, prespecified feature offset in the stress experiment is a distribution-shift intervention on the simulator. It is not automatically a pure covariate shift: changing observed features while retaining latent labels can also change P(Y | X). A result under this intervention does not establish robustness to all laboratory batch effects.

## Split conformal specification

For each calibration specimen, let s_i = 1 - p_hat(Y_i | mean(X_i)). With n calibration specimens and error level alpha, let k = ceil((n + 1)(1 - alpha)). Use the k-th smallest score if k <= n, and positive infinity otherwise. Return the label set C(x) = {y in {0, 1}: 1 - p_hat(y | x) <= q}. Ties are included. Do not interpolate the order statistic or force an empty set to contain the most likely label.

Under the stated exchangeability conditions, this construction provides marginal coverage of at least 1 - alpha for a new specimen. It does not guarantee coverage conditional on every feature value, label, or subgroup, and does not make the neural-network probabilities calibrated. See [Angelopoulos and Bates, A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511).

The set may be empty, a singleton, or {0, 1}. All three outcomes must be retained and reported. A method that always returns both labels has perfect coverage and no label discrimination. Coverage therefore needs to be evaluated alongside set size and point-prediction quality.

## Prespecified questions and ways to falsify them

These are proposed hypotheses for a larger study. The three-seed smoke experiment cannot resolve them.

| Question | Proposed contrast | Evidence against the anticipated explanation |
|---|---|---|
| H1: Does repeated-measure leakage produce apparent skill without label signal? | Specimen-permuted labels; nearest neighbor with row splits versus whole-specimen splits, varying repeat correlation and repeat count. | No reproducible optimism gap in the prespecified high-correlation regime. Inspect whether the negative-control construction is sufficiently identifiable; do not tune the demonstration until it looks dramatic. |
| H2: Does the neural model improve prediction and informative set size beyond simple alternatives? | Paired MLP-minus-baseline Brier score and mean-set-size differences for both linear and interaction logistic regression; include nonlinear, linear, and null regimes. | No reproducible Brier improvement or set-size benefit at comparable coverage; worse performance in linear/small-sample regimes is an informative limitation. |
| H3: Does ordinary conformal calibration lose reliability under some shifts? | Coverage and efficiency curves across prespecified shift strength, measurement noise, and signal-to-noise conditions. | Coverage remains satisfactory in the selected shift regimes; conclude that those regimes did not expose failure, not that arbitrary shift is safe. |
| H4: At a fixed measurement budget, when are more specimens preferable to more technical repeats? | A future factorial experiment with total measurement budget B approximately equal to n times m, trading n for m. | No stable regime in which the preferred allocation changes, or a simple analytic/noise baseline explains the observed pattern entirely. |

For H2, the MLP need not win. The interaction-logistic baseline is especially important when the simulator includes interactions: a comparison only against a linear model would confound neural-network value with access to a nonlinear feature map. Prevalence and full-label-set baselines are also scientifically useful. A benchmark with only a nonlinear generator cannot support a general superiority claim.

## Evaluation and uncertainty

Report each model, seed, and scenario separately: specimen count; coverage; mean set size; empty-set and singleton rates; accuracy; Brier score; and log loss. A reliability summary such as ECE is descriptive and sensitive to the chosen bins. Where a subgroup analysis is prespecified, report its denominator and uncertainty, including when it is too small to interpret.

The v0.1 Wilson interval for coverage describes binomial sampling uncertainty over independent held-out specimens **conditional on the fitted model and realized calibration set**. It does not quantify the variability from collecting new training or calibration data, and its lower bound need not exceed the conformal nominal target in every run. A low observed coverage value in one small test sample is not alone a proof that the conformal algorithm is incorrect.

Within a seed and a fixed scenario, all three models predict the same held-out specimens. Resample those specimens jointly when bootstrapping each MLP-minus-baseline Brier difference; keep paired predictions together. This bootstrap also conditions on the realized fits and calibration. It is not an interval for the full train-and-calibrate procedure. The two shift settings reuse latent specimens, so comparisons across them are paired and are not independent replications.

Across seeds, summarize complete-run metrics and retain their distribution. An MCSE from only three runs is an unstable smoke-test summary. Do not describe it as a reliable estimate of a method's expected performance, publish p-values for superiority from it, or treat all test rows from all fits as independent Bernoulli trials.

## Larger simulation design and precision planning

Before a larger run, freeze the estimand, generator grid, primary contrasts, model-selection budget, failure rules, seeds, and stopping rule in a versioned configuration. Seek approval for an expensive benchmark before running it. Proposed design factors include independent specimen count, repeat count, within-specimen noise, nonlinear signal, dimension, class prevalence, fixed feature shift, and later informative repeat count. Add shared-batch dependence only with an explicit supercluster sampling design.

Use a two-stage design:

1. Run a modest, separately labeled pilot of independent complete datasets to estimate runtime and the standard deviation of primary run-level metrics. Select final replication counts using variance and a predefined precision target, not the most favorable observed effect. Exclude pilot runs from a confirmatory analysis, or predefine their use before inspection.
2. For R independent complete runs, compute metric mean C_bar and MCSE = sd(C_r) / sqrt(R). This accounts for variation across newly generated training, calibration, and test data. A rough precision target h for a 95% interval gives R approximately (1.96 times sd(C_r) / h)^2, rounded upward. For illustration only, a pilot SD of 0.05 and h = 0.01 would require about 97 runs; the actual SD has not been measured.

For a simpler design with exactly one fresh test specimen per independently refitted procedure, coverage indicators are independent Bernoulli outcomes. Planning around coverage 0.90 and a 95% half-width of 0.01 requires approximately 3,458 complete runs. This is a planning approximation, not a prescription for this project. Many test specimens per fit reduce conditional test noise but do not remove calibration-to-calibration variability.

For a prespecified paired superiority contrast with minimally meaningful effect delta, estimate the SD of complete-run paired differences D_r in the pilot. A normal planning approximation for two-sided 5% size and 80% power is R approximately ((1.96 + 0.84) times sd(D_r) / delta)^2. For a hypothetical SD of 0.15 and delta = 0.05 in the same metric units, this is about 71 runs. Neither value is an observed result. Final testing must account for the number of primary comparisons; exploratory grid searches should be reported as exploratory. Simulation runs are not biological replicates, and simulation power is not clinical study power.

A future comparison of set efficiency should first define acceptable coverage, for example a prespecified noninferiority margin on marginal coverage with adequate precision. Selecting the smallest set only among methods that happen to pass a noisy three-run coverage threshold creates selection bias. Report the coverage–efficiency tradeoff for all methods.

## Baselines and ablations before a research claim

| Stage | Required comparison | Purpose |
|---|---|---|
| v0.1 | Linear logistic regression, interaction logistic regression, and two-hidden-layer MLP on identical specimen means and splits | Compare a learned nonlinear representation with both a linear model and a fixed nonlinear feature map; keep calibration identical. |
| v0.1 | Permuted-label nearest-neighbor row split versus specimen split | Demonstrate a failure mechanism without invoking biological label signal. |
| Next benchmark | Prevalence-only classifier, logistic model, an approved tree-based baseline, and full-label prediction sets | Avoid weak-model and coverage-only comparisons. |
| Next benchmark | One repeat, mean, mean-plus-variance/count, and a trained permutation-invariant set encoder | Determine whether a learned encoder adds value beyond simple summaries. |
| Next benchmark | Linear, nonlinear, and null generators; low and high measurement noise | Check whether conclusions depend on a network-friendly generator. |
| Method development | Relevant hierarchical/shift-aware conformal methods with aligned prediction targets | Compare against existing inference methods before claiming a new one. |

The negative control uses a different model and intentionally invalid split to isolate leakage. It is not an additional fair competitor in the three-model accuracy ranking. The permuted label must be assigned once per specimen and copied to its measurement rows; independent row-wise permutation would destroy the structure being tested.

DeepSets or another set encoder is a future extension, not part of v0.1. PyTorch, MPS, or additional packages require the repository's installation and institutional approval process. First verify NumPy forward passes and gradients against small analytic or finite-difference examples. Future hardware speed comparisons must use the same task and accuracy criterion.

## Existing work and a credible contribution gate

This is a focused literature starting point, checked on 2026-10-22; it is not an exhaustive novelty search.

- [Dunn, Wasserman, and Ramdas, Distribution-Free Prediction Sets for Two-Layer Hierarchical Models](https://arxiv.org/abs/1809.07441): develops hierarchical prediction methods, including pooling and subsampling approaches. Repeated measurements and hierarchical conformal validity are established topics.
- [Lee, Barber, and Willett, Distribution-free inference with hierarchical data](https://arxiv.org/abs/2306.06342): addresses distribution-free inference for hierarchical observations. Its target and exchangeability assumptions must be aligned before it becomes a comparator to specimen-label prediction.
- [Mallick, Tchetgen Tchetgen, Dobriban, and Lee, Generalized Hierarchical Conformal Prediction](https://arxiv.org/abs/2608.15500): an August 2026 preprint addressing prediction after observations from a test group are available. Its stated assumptions include reference-group-size ignorability. It is related future reading, not an implemented baseline or evidence of novelty here.
- [Tibshirani, Barber, Candès, and Ramdas, Conformal Prediction Under Covariate Shift](https://arxiv.org/abs/1904.06019): weighted conformal methods address a specified shift structure with a density ratio. This does not license a guarantee for an arbitrary feature-offset stress test.
- [Zaheer et al., Deep Sets](https://arxiv.org/abs/1703.06114): relevant to a future permutation-invariant representation of repeated observations. Learning a representation and proving calibration validity are separate tasks.

An attainable first contribution is a reusable, documented benchmark of the tradeoff between independent specimen sampling and repeated measurement under fixed cost, with explicit uncertainty and failure regimes. Its uniqueness is unverified until a broader literature and software comparison is completed.

A possible later methodological question is whether a repeat-aware score using measurement variability and repeat count improves prediction-set efficiency while preserving a precisely stated specimen-level coverage guarantee under a fixed collection protocol. A learned score alone does not require a new conformal theorem if the entire specimen object remains exchangeable. Adaptive repeat collection, test-time adaptation, or informative cluster sampling can change the target and symmetry; any claimed guarantee then needs its own proof or an applicable existing result. Do not rename a known construction as a new method.

Move from prototype to a methodological claim only after all of the following are available:

1. A precise target and a gap documented against relevant existing methods and software.
2. A frozen implementation, sufficient complete-run Monte Carlo precision, strong baselines, and reported failure cases.
3. A proof for any new guarantee, with assumptions that match the implemented sampling and collection protocol, independently reviewed by someone familiar with conformal inference.
4. For any claim of real-world transferability, an independent dataset whose consent, ownership, license, and publication permissions are explicitly established. A theory contribution can stand on a proof and careful simulation, but synthetic results alone cannot establish laboratory utility.
5. A public report that distinguishes prediction uncertainty, parameter inference, and causal conclusions, and that includes negative or inconclusive findings.

## Prediction coverage is not parameter or causal inference

The v0.1 set covers a future label under stated sampling conditions. It is not a confidence interval for a regression coefficient, an effect of a biological modification, or a causal treatment effect. Neural network weights do not have a unique scientific interpretation. Bootstrap intervals for predictive performance address a further, explicitly conditional estimation problem.

If the project later includes an effect-estimation question, define a separate estimand and observation model. For example, a population mean contrast in a known-truth simulator could be estimated using an appropriate repeated-measures or hierarchical model, with bias and interval coverage checked across independent datasets. A causal interpretation would additionally require an identification design and assumptions; prediction performance cannot supply them.

## Minimum evidence for release

- Split-integrity and training-only preprocessing checks; conformal finite-sample order-statistic, tie, and insufficient-calibration tests; neural-network gradient checks; deterministic small-run reproducibility.
- Machine-readable configuration, seeds, environment versions, sample denominators, raw per-run metrics, and a clear distinction between generated data and external data.
- README and result captions identifying the three-seed run as a smoke experiment, the feature shift as a stress test, and the limits of conditional intervals.
- No claims of biological discovery, new statistical theory, population-level model superiority, or learner mastery based on this release.

The strongest early portfolio signal is that the code makes an inferential mistake easy to detect and the report says exactly what its evidence supports.
