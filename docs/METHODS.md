# ReplicateShift v0.2 methods

The current implementation has two entry points: a CSV model toolkit and an executed measurement-allocation experiment. Both predict one binary label per new independent specimen. The [v0.1 methods](METHODS_V0_1.md) describe the separate, preserved smoke study.

## Prediction target and independence

A specimen contributes a set of technical measurements, one label, one training-loss contribution, and one calibration score. Repeats are not additional independent specimens. The implementation assumes independence between specimens; it does not supply a patient/family/batch-level sampling scheme. Existing-ID rejection catches direct reuse, not renamed measurements or unrecorded dependencies.

## Synthetic observation model

For independent specimen i, draw three latent signal variables z_i ~ N(0,I). Draw a binary label with log odds `2.5*z_i0*z_i1 + 1.2*z_i2`. Append nine independent nuisance features with SD 1.5. Repeat j observes the same latent vector plus independent Gaussian noise with SD σ. The allocation pilot evaluates σ = 0.35 and 1.0 with paired underlying normal draws. Labels are drawn once per specimen.

This is an omics-inspired repeated-measurement abstraction. It is not a physical or biological model of RNA sequencing, mass, retention time, modification state, or an actual instrument. There is no shared random batch effect. Under this additive Gaussian mechanism, the repeat mean contains the relevant latent-location information; within-specimen SD is largely ancillary. A neural advantage is not assumed.

## Fixed-budget allocation pilot

The development budget is 600 equal-cost assay rows. The alternatives m = 1, 3, 6 fund respectively 600, 200, 100 independent development specimens. Of the assay budget, 60% funds fitting, 20% validation, and 20% calibration. Thus role counts are `360/m`, `120/m`, and `120/m`. Each policy is evaluated on 300 additional specimens with m measurements each, costing `300*m` extra assay rows. Procurement, preparation, failed measurements, and computation are excluded from the cost model.

Repeat count changes both the development allocation and test-time measurement policy. This compares complete policies, not an isolated training intervention. Calibration sizes 120, 40, and 20 also change the resolution and variability of the conformal threshold.

For each root seed, independent SeedSequence children generate maximum-sized fitting, validation, calibration, and test pools, plus model initializations. Allocations use nested prefixes of the appropriate role pool and of its repeats. A specimen cannot switch roles across policies. Model/scenario comparisons within a root seed are paired; different root seeds generate independent complete experiments. Each model has its own initialization stream, so redundant SD features at m=1 need not yield identical finite-epoch predictions across baseline fits.

The frozen [configuration](../configs/allocation_pilot.json) uses seeds 101, 211, 307, 401, 503, 601, 701, 809, 907, 1009. These are ten exploratory complete-run replicates, not biological replicates or a confirmatory sample size. No hyperparameters were retuned after viewing the pilot test results.

## Preprocessing and model fitting

All transformations fit training specimens only. The statistical baselines first form specimen means and all 66 unordered pairwise products, then standardize each resulting feature with training mean and population SD. The second baseline additionally includes the 12 within-specimen population SDs (ddof=0), defined as zero for singleton specimens. Constant training features use scale one.

Deep Sets standardizes raw repeat features using equal total weight per specimen: each repeat's weight within a specimen is 1/m_i, and each specimen receives weight 1/n. Its masked encoder is

`p_i = sigmoid(W4 * tanh(W3 * mean_j(phi(x_ij)) + b3) + b4)`

where phi has two tanh layers with widths 16 and 12, and the decoder hidden width is 12. Padding contributes neither to scaling nor pooling nor loss. Repeat permutation leaves the score unchanged. Mean pooling also makes duplicating an entire set invariant, so cardinality is not a separate feature. Ragged-input support should not be interpreted as a learned model of evidence quantity.

All methods minimize specimen-mean binary cross-entropy plus `0.5*0.003*sum(weight²)`, excluding biases from the penalty. Full-batch Adam uses beta1=.9, beta2=.999, epsilon=1e-8. Logistic learning rate is .03; Deep Sets uses .003. Every model trains 120 epochs and restores the checkpoint with lowest unpenalized validation BCE after an update; ties retain the earliest checkpoint. Validation affects selection only. Calibration and test labels never alter fitted weights or preprocessing.

The NumPy implementation is explicit so its gradients and state can be inspected and tested. It does not require PyTorch, a GPU, or downloaded weights. This release makes no large-data throughput claim.

## Split conformal prediction

With a selected, fixed predictor, each calibration specimen has score `s_i = 1 - p_hat(y_i | X_i)`. Set `k = ceil((n_cal+1)*(1-alpha))`. The threshold q is the kth ordered score, or positive infinity if k > n_cal. Include candidate label y exactly when `1-p_hat(y | X) <= q`. No interpolation is used; ties are conservative. At alpha=.1, the pilot ranks are 109/120, 37/40, and 19/20.

If calibration and future specimens are exchangeable as complete objects, and score fitting/selection is independent of them, marginal label coverage is at least 1-alpha. This averages over calibration and new-test sampling. It does not promise coverage for a particular fixed calibration set, every class or subgroup, or an arbitrary new measurement policy. Whole-specimen exchangeability can allow dependent repeats within a specimen, but the training/scoring implementation cannot repair dependencies between specimens. Calibration guarantees prediction-set coverage, not probability calibration or parameter/causal inference.

## Metrics and Monte Carlo uncertainty

Per-run outputs retain accuracy, Brier score, clipped log loss, descriptive ten-bin ECE, coverage, set size, and empty/singleton rates. Coverage Wilson intervals in per-run metrics are conditional on the fitted model and realized calibration threshold. They do not include fitting/calibration variability and are not the uncertainty summary for comparisons across complete runs.

For each model/scenario, report the mean metric over ten complete runs and MCSE = sample SD / sqrt(10). This estimates simulation precision across freshly drawn development and test data. It is not a biological confidence interval. For each baseline, calculate Deep-Sets-minus-baseline Brier within each root seed, then report the mean difference and MCSE of those ten paired differences. Never treat 54,000 prediction rows as 54,000 independent model-development experiments. No p-values or multiplicity-adjusted superiority claims are made. Coverage below 0.90 in a small empirical run average alone does not refute the marginal theorem.

## CSV toolkit and reusable model

The CSV workflow randomly partitions complete input specimens into 50/15/15/20% fitting/validation/calibration/holdout roles without label stratification. It requires at least 60 specimens and a labeled binary input. This workflow differs from the controlled pilot's budget allocation and external test pool. The model/scaler architecture and optimization defaults match the pilot; the number of features follows the CSV schema.

Required identifiers are `specimen_id` and `replicate_id`; at least one finite numeric `feature_*` column is required. Fitting also requires `label`, consistently 0 or 1 within each specimen. Repeat counts may vary. Prediction labels are optional. Feature order can change, but feature names must match the fitted schema exactly. There is no automatic missingness imputation, batch correction, or higher-level grouping.

The fit output includes JSON metadata, NPZ weights loaded with `allow_pickle=False`, holdout predictions, and holdout metrics. Metadata records input SHA-256, feature schema, software versions, model configuration, training histories, separate role seeds/IDs, and the frozen conformal threshold. NPZ stores model parameters and fitted scaling arrays. The weights file has a checked SHA-256. Stored IDs are plaintext for overlap checking; this release's examples contain synthetic IDs only. New prediction IDs must not appear in any previous role. Existing output paths are refused.

## Provenance and limitations

The experiment writes its protocol before fitting, fingerprints the source before/after fitting, and saves raw compressed predictions. The manifest records source/output hashes, ordered role-pool IDs and hashes, role seeds, environment, runtime, and traced allocation peak. Per-run counts reconstruct the nested prefix splits. Runtime excludes final report serialization; tracemalloc is not total resident or device memory. The HTML report is a self-contained results table.

Historical v0.1 manifests refer to historical source, not necessarily the current v0.2 working tree. Their output files remain unchanged. Reproduction uses the matching recorded revision and environment where exact floating-point identity is required.

The current contribution is reusable research software and an explicit measurement-allocation benchmark. Real-data utility, robustness across other observation mechanisms, optimal allocation, new architecture, and new coverage theory remain unestablished.

Sources: [Deep Sets](https://arxiv.org/abs/1703.06114); [Angelopoulos & Bates](https://arxiv.org/abs/2107.07511); [Dunn, Wasserman & Ramdas](https://arxiv.org/abs/1809.07441).
