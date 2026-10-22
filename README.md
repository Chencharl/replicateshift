# ReplicateShift

**Deep learning can predict a specimen. Can we trust its uncertainty?**

ReplicateShift is a personal research-engineering project connecting deep learning, biostatistics, and reproducible omics research. It studies a practical distinction: repeated measurements of one specimen do not create additional independent specimens.

The first release is a working, synthetic benchmark. It compares a two-hidden-layer neural network with linear and interaction-based logistic regression, then constructs prediction sets for **new specimens**. A separate negative control reveals how technical-repeat leakage can make unrelated labels appear predictable.

**Current status:** v0.1 research prototype; 28 automated tests pass locally. Three simulation seeds have been executed. No laboratory or patient data are included. A new statistical method, real-world validation, and independent learner mastery are not claimed.

## Why this project

My research interests connect biostatistics with computational RNA analysis. This project explores the boundary between a model that predicts well and an inference procedure whose uncertainty is defensible. The practical questions are:

1. What changes when the independent unit is a specimen rather than an assay row?
2. Can prediction sets retain useful coverage without routinely including both labels?
3. How do measurement shifts change accuracy, probability quality, and coverage?
4. When does a simple interaction model outperform a neural network?

The distinctive direction is a **replicate-allocation benchmark**: at a fixed measurement budget, when should a study measure more specimens versus more repeats? The current release establishes the foundations; [the research plan](docs/RESEARCH_PLAN.md) specifies the experiments needed to make that contribution credible. These questions build on established conformal-inference literature rather than claiming the grouping principle is new.

## First executed result

600 synthetic specimens per seed, three technical repeats each, and three seeds. Training, calibration, and testing use disjoint specimens. These are descriptive smoke-study means, not evidence of general model superiority.

| Model | Unshifted accuracy | Unshifted Brier ↓ | Unshifted coverage | Shifted coverage |
|---|---:|---:|---:|---:|
| Linear logistic | 0.683 | 0.207 | 0.928 | 0.847 |
| Interaction logistic | 0.750 | 0.190 | 0.908 | 0.833 |
| Neural network | 0.653 | 0.297 | 0.900 | 0.847 |

The neural network did **not** outperform the simpler nonlinear baseline in this initial run. Its near-target unshifted coverage did not imply good probability calibration or superior predictions. Under the specified assay shift, all three empirical coverage means fell below the nominal 90% target; the exchangeability-based guarantee does not apply to this shifted setting.

In the separate nearest-neighbor negative control, labels were permuted across specimens. Row splitting still gave **90.0–91.1%** accuracy, while splitting by specimen gave **47.5–50.8%**. The former often predicts another repeat of a specimen already seen during training. This diagnostic demonstrates leakage in this simulator, not a universal effect size for neural networks.

Read the [complete executed report](results/smoke_v0.1/REPORT.md) for set sizes, per-seed results, conditional uncertainty intervals, and limitations. All predictions, split identifiers, configurations, and hashes are included.

## Run it

Python 3.10+ and NumPy are sufficient. No pretrained weights, external datasets, GPU, service, or account is required. Use an environment where package installation is permitted.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m replicateshift.benchmark --output results/local-demo
```

If NumPy is already available, run without installing the package:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m replicateshift.benchmark --output results/local-demo
```

The default smoke study uses seeds `7 19 43`, 600 specimens, three repeats, 180 training epochs, assay shifts `0 1.5`, and miscoverage level `alpha=0.1`. Use a fresh output directory: existing results are never overwritten. The [manifest](results/smoke_v0.1/manifest.json) records the actually executed software versions and configuration; floating-point results may vary across platforms or NumPy versions.

## What is implemented

- **Synthetic repeated assays:** specimen-specific latent signals and nuisance fingerprints, independent technical noise, nonlinear binary labels, and a paired measurement-shift stress test.
- **Deep learning:** a `12 → 32 → 16 → 1` tanh network, stable binary cross-entropy, explicit backpropagation, L2 regularization, and Adam in NumPy. Every parameter's analytic gradient is checked numerically on a small fixture.
- **Baselines:** linear logistic regression and logistic regression with all 66 pairwise interactions; each has its own training-only standardizer.
- **Predictive inference:** specimen-level split conformal sets with a finite-sample order statistic and conservative tie handling.
- **Evaluation:** accuracy, Brier score, log loss, descriptive calibration error, coverage, set size, empty/singleton rates, and paired specimen-bootstrap Brier differences.
- **Reproducibility:** separated random streams, explicit split manifests, source/data/output hashes, deterministic tests, and GitHub Actions.

The network consumes specimen-mean features. A learned set encoder such as Deep Sets is a future extension, not part of v0.1. The project does not implement causal-effect estimation, sequence reconstruction, or a validated LC-MS observation model.

## Statistical contract

One specimen supplies one label and one calibration score. With a fixed predictor trained independently, exchangeable calibration and new test specimens support **marginal** prediction-set coverage. This is neither a guarantee for every specimen nor a confidence interval for a biological effect. Shared batch effects, adaptive test reuse, and distribution shift require additional assumptions or methods.

Coverage is reported with set size because returning both labels always covers the answer. Test-set Wilson and bootstrap intervals are conditional on the fitted models/calibration set; they do not include the entire model-development process. Three seeds are insufficient for a publication-level simulation study.

See [methods](docs/METHODS.md), [research plan](docs/RESEARCH_PLAN.md), [code review](docs/CODE_REVIEW.md), and [validation record](docs/VALIDATION.md).

## Build understanding alongside the code

The [learning plan](docs/LEARNING_PLAN.md) starts with a 45–60 minute split-leakage exercise and develops through three weekly contracts at 10–12 hours per week. Completion is evidenced by explanations, tests, and experiments—not the presence of AI-generated code.

Project owner: **Chen Yang**. Initial implementation, documentation, and review were developed with AI assistance. Independent understanding and any future scientific claims must be demonstrated separately.

## References

- Angelopoulos & Bates. [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511).
- Dunn, Wasserman & Ramdas. [Distribution-Free Prediction Sets for Two-Layer Hierarchical Models](https://arxiv.org/abs/1809.07441).
- Zaheer et al. [Deep Sets](https://papers.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html), a future architecture reference.

MIT license. Proposed methodological extensions and contributions are welcome; preserve the estimand, independent unit, negative controls, and reproducible evidence when changing the benchmark.
