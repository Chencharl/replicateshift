# ReplicateShift — measurement-allocation pilot

Executed exploratory simulation. This evaluates complete development/deployment policies; it is not a biological result or an optimal laboratory allocation recommendation.

Development budget: **600 assay rows** in every condition. Independent test specimens: 300 per condition, requiring an additional `test_specimens × repeats` assays. Each condition uses 10 independent complete-run seeds.

## Results

Mean ± Monte Carlo standard error across independent runs. MCSE estimates simulation precision, not a biological confidence interval. All methods and scenarios within each seed are paired.

| Noise SD | Repeats | Dev. specimens | Model | Brier ↓ | Coverage | Mean set size | Test assays |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0.35 | 1 | 600 | deep_set | 0.230 ± 0.005 | 0.905 ± 0.008 | 1.662 ± 0.025 | 300 |
| 0.35 | 1 | 600 | mean_interaction | 0.226 ± 0.008 | 0.906 ± 0.009 | 1.619 ± 0.047 | 300 |
| 0.35 | 1 | 600 | mean_sd_interaction | 0.226 ± 0.007 | 0.899 ± 0.010 | 1.602 ± 0.041 | 300 |
| 0.35 | 3 | 200 | deep_set | 0.248 ± 0.005 | 0.889 ± 0.011 | 1.690 ± 0.034 | 900 |
| 0.35 | 3 | 200 | mean_interaction | 0.247 ± 0.002 | 0.907 ± 0.014 | 1.761 ± 0.033 | 900 |
| 0.35 | 3 | 200 | mean_sd_interaction | 0.246 ± 0.001 | 0.910 ± 0.016 | 1.772 ± 0.035 | 900 |
| 0.35 | 6 | 100 | deep_set | 0.256 ± 0.006 | 0.883 ± 0.023 | 1.727 ± 0.047 | 1800 |
| 0.35 | 6 | 100 | mean_interaction | 0.255 ± 0.008 | 0.914 ± 0.014 | 1.778 ± 0.042 | 1800 |
| 0.35 | 6 | 100 | mean_sd_interaction | 0.252 ± 0.004 | 0.878 ± 0.029 | 1.696 ± 0.074 | 1800 |
| 1 | 1 | 600 | deep_set | 0.247 ± 0.002 | 0.907 ± 0.004 | 1.745 ± 0.010 | 300 |
| 1 | 1 | 600 | mean_interaction | 0.252 ± 0.003 | 0.901 ± 0.005 | 1.757 ± 0.014 | 300 |
| 1 | 1 | 600 | mean_sd_interaction | 0.252 ± 0.004 | 0.907 ± 0.005 | 1.782 ± 0.014 | 300 |
| 1 | 3 | 200 | deep_set | 0.250 ± 0.005 | 0.894 ± 0.011 | 1.721 ± 0.028 | 900 |
| 1 | 3 | 200 | mean_interaction | 0.250 ± 0.002 | 0.916 ± 0.011 | 1.804 ± 0.021 | 900 |
| 1 | 3 | 200 | mean_sd_interaction | 0.248 ± 0.001 | 0.934 ± 0.012 | 1.851 ± 0.025 | 900 |
| 1 | 6 | 100 | deep_set | 0.262 ± 0.008 | 0.909 ± 0.014 | 1.785 ± 0.033 | 1800 |
| 1 | 6 | 100 | mean_interaction | 0.250 ± 0.003 | 0.882 ± 0.019 | 1.701 ± 0.046 | 1800 |
| 1 | 6 | 100 | mean_sd_interaction | 0.252 ± 0.004 | 0.902 ± 0.016 | 1.751 ± 0.048 | 1800 |

## Paired model differences

Negative differences favor Deep Sets. These differences use each seed's complete-run Brier metrics; individual rows from different fitted procedures are not pooled as independent trials.

| Noise SD | Repeats | Baseline | Deep Sets minus baseline Brier | MCSE of paired difference |
|---:|---:|---|---:|---:|
| 0.35 | 1 | mean_interaction | 0.0044 | 0.0049 |
| 0.35 | 1 | mean_sd_interaction | 0.0046 | 0.0048 |
| 0.35 | 3 | mean_interaction | 0.0013 | 0.0051 |
| 0.35 | 3 | mean_sd_interaction | 0.0021 | 0.0048 |
| 0.35 | 6 | mean_interaction | 0.0012 | 0.0117 |
| 0.35 | 6 | mean_sd_interaction | 0.0044 | 0.0081 |
| 1 | 1 | mean_interaction | -0.0053 | 0.0015 |
| 1 | 1 | mean_sd_interaction | -0.0059 | 0.0025 |
| 1 | 3 | mean_interaction | -0.0001 | 0.0062 |
| 1 | 3 | mean_sd_interaction | 0.0018 | 0.0050 |
| 1 | 6 | mean_interaction | 0.0125 | 0.0098 |
| 1 | 6 | mean_sd_interaction | 0.0104 | 0.0100 |

## Experimental contract

- Separate maximum pools for training, validation, calibration, and testing prevent any specimen from changing roles across allocations. Prefix subsets and paired noise draws make policy comparisons within a seed paired.
- 60/20/20% of the development assays fund training/validation/calibration respectively. Calibration sample size and finite-sample quantile resolution therefore change with repeat count.
- All trainable methods select the minimum validation-BCE checkpoint from a fixed training budget. Scaling uses training specimens only; calibration/test labels never select checkpoints.
- The two logistic baselines share all mean features and pairwise mean interactions. The second adds within-specimen SD, explicitly zero for one repeat. Deep Sets learns an embedding from the individual repeats and uses masked mean pooling.
- The simulator has independent specimens and conditionally independent Gaussian technical noise. Sample SD is largely ancillary here; no neural-model advantage is assumed. This is not a calibrated model of RNA chemistry or an LC-MS instrument.
- Nominal conformal coverage is 90% under matching exchangeable calibration/test specimen policies. The guarantee is marginal, not conditional on every specimen, subgroup, or fixed calibration realization.
- Each repeat count changes both the development allocation and the test-time measurement policy. Test assay costs are additional and explicitly reported. The study does not isolate a training-only effect.
- The budget counts equal-cost assay rows only; specimen procurement, preparation, failed measurements, and computing costs are excluded.
- Ten seeds by default provide a small pilot, not a powered confirmatory comparison. Selecting the best-looking allocation from this table is exploratory. No p-values, universal optimum, new statistical theorem, or real-data benefit are claimed.

## Reproduce and inspect

Replace `path/to/this/run` with the directory containing this report, then run from the repository root:

```sh
PYTHONPATH=src python -m replicateshift.allocation --config path/to/this/run/protocol.json --output results/local-allocation
```

`protocol.json` contains the executed configuration. `per_run_metrics.csv` records selected epochs and role-specific specimen counts; together with the manifest's ordered pool IDs these determine each prefix split. `summary.csv`, `paired_comparisons.csv`, and `predictions.csv.gz` retain the remaining results. `manifest.json` records role seeds, pool hashes, source/output hashes, runtime, and traced-memory scope. `report.html` is a self-contained filterable table with no external scripts or services.
