# Methods and claim boundaries

## Estimand and independent unit

The prediction target is one binary label for a previously unseen specimen, given all its technical measurements. It is not a new assay row from a known specimen, an RNA identity, a biological effect, or a causal treatment contrast. The specimen is the highest independent sampling unit in v0.1. There are no shared random batch effects in the simulator.

## Generative model

For specimen i, draw z_i ~ N(0,I_3). Draw its binary label from a Bernoulli distribution with log odds `2.5*z_i0*z_i1 + 1.2*z_i2`. Append nine independent nuisance features with SD 1.5. All measurements of that specimen share these latent features. Add independent N(0,0.35²) noise to every feature in every technical repeat. The standard configuration contains 600 specimens and three repeats each.

For a measurement shift of s, transform the first three latent observed features to `(z0+s, z1/(1+abs(s)), z2+0.5*s)` before adding the same paired technical noise. Labels and nuisance features stay fixed. This is a controlled assay perturbation; it need not preserve P(Y|X). The data are not a calibrated simulation of mass, retention time, RNA modifications, or an actual instrument.

## Splits, preprocessing, and models

Randomly allocate complete specimens into 60% training, 20% calibration, and 20% testing without label stratification. Average the technical measurements within each specimen. Fit all preprocessing on training specimens only. Keep calibration labels exclusive to conformal scoring, and test labels exclusive to evaluation. No hyperparameter selection or early stopping uses either set.

Three models use mean binary cross-entropy plus `0.5*0.001*sum(weights²)` with unpenalized intercepts. All run 180 full-batch Adam steps (`beta1=.9`, `beta2=.999`, `epsilon=1e-8`).

| Model | Features / layers | Learning rate |
|---|---|---:|
| Linear logistic | 12 specimen-mean features | .03 |
| Interaction logistic | 12 main effects and 66 pairwise products | .03 |
| MLP | 12 → 32 tanh → 16 tanh → 1 sigmoid | .01 |

Feature augmentation precedes training-only mean/SD standardization for the interaction model. The interaction map uses every unordered feature pair, without selecting the true generating interaction. These fixed hyperparameters are transparent baselines, not a claim of optimal tuning. Future comparisons require an inner training validation split and a fresh final test design.

SeedSequence-spawned streams separate the simulator, partition, each model's initialization, bootstrap, and negative control. Scenario shifts intentionally share the simulation stream to preserve paired latent test specimens. Root seeds are 7, 19, and 43; exact role seeds are saved in each manifest.

## Split conformal sets

For each calibration specimen, use nonconformity `s_i = 1 - p_hat(y_i | x_i)`. Let `k = ceil((n_cal+1)*(1-alpha))`. The threshold is the kth smallest calibration score; if k exceeds n_cal, use positive infinity. Include label y when `1-p_hat(y|x) <= threshold`. Interpolation is not used, and ties are included conservatively.

For exchangeable calibration and test specimens and an independently trained fixed scoring rule, the prediction set has marginal coverage at least `1-alpha`. This averages over the calibration and new-test sampling; it does not promise coverage for each fixed calibration realization, class, covariate stratum, specimen, or shifted population. The target here is a whole-specimen label; no claim is made to implement hierarchical conformal methods for a new observation within a new or partially observed cluster.

## Evaluation and uncertainty

Evaluate each model on exactly the same test specimen identities and labels. Report accuracy, mean squared probability error (Brier), clipped log loss, and a descriptive ten-bin positive-class expected calibration error. Also report conformal coverage, mean set size, empty-set rate, and singleton rate. Avoid interpreting ECE as a formal inferential test.

Coverage Wilson intervals use the number of independent test specimens, conditional on the fitted model and realized calibration threshold. Paired bootstrap comparisons resample test specimens with replacement 1,000 times, taking the 2.5th and 97.5th percentiles of the MLP-minus-baseline Brier difference. These intervals do not incorporate model fitting, calibration-set variability, or multiplicity across comparisons. Repeated-run means and MCSE are recorded separately; three seeds are only a smoke check.

The shifted and unshifted scenarios are paired sensitivity conditions. Their predictions must not be pooled as independent measurements. There is no shifted-data coverage theorem asserted here.

## Negative control

Permute labels across whole specimens, then assign that permuted label to every repeat within each specimen. Train one-nearest-neighbor on individual rows under either a random row split or a specimen-disjoint split. Report test-row accuracy and the fraction of test rows whose specimen appears in training. The row split often asks an easier, different question: another measurement of a specimen already observed. The negative control checks memorization through specimen fingerprints, not neural model performance or probability calibration.

## Extension rules

If adding shared batches or donors above specimens, split and resample the highest independent units or adopt and validate a corresponding hierarchical inference method. If using variable repeat counts, explicitly define the prediction target and how repeat count relates to outcomes and uncertainty. Do not treat more assay rows as more biological samples. See the research plan for preregistered study axes and contribution gates.

Sources: [Angelopoulos & Bates](https://arxiv.org/abs/2107.07511); [Dunn, Wasserman & Ramdas](https://arxiv.org/abs/1809.07441).
