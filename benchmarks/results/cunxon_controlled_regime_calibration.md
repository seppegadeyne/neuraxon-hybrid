# cuNxon controlled-regime criticality calibration

Status: `controlled-regime calibration completed`
Hypothesis: `controlled_regime_calibration`
Source issue: https://github.com/sisutuulenisa/neuraxon-hybrid/issues/86

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Modes: infer, train
- Seed offsets: 133, 134
- Steps / sample interval: 128 / 8
- Samples: 288

## Why this probe exists

This report follows Qubic NIA Vol. 8 by calibrating the snapshot branching/avalanche estimator under controlled low, medium and high input-drive regimes. The goal is estimator calibration, not an intelligence claim; stress/control quality must beat constant baselines before runtime criticality is treated as useful computation evidence.

## Controlled regimes

| Regime | Drive scale | Samples | Mean branching | Branching range | Mean active ratio | Mean neutral | Entropy | Actions | Splits beating constants |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| low-drive | 0.25 | 96 | 0.030866 | 0.000000..0.357490 | 0.990481 | 0.978481 | 0.274261 | assertive=2, execute=2, query=90, retry=2 | train |
| medium-drive | 1.00 | 96 | 0.086529 | 0.000000..0.357490 | 0.970356 | 0.925696 | 0.846045 | assertive=4, execute=11, query=78, retry=3 | hard_holdout, holdout, train |
| high-drive | 2.00 | 96 | 0.128423 | 0.000000..0.357490 | 0.956839 | 0.873544 | 1.374276 | assertive=2, execute=22, query=69, retry=3 | hard_holdout, holdout, train |

## Split accuracy versus constant baselines

| Split | Accuracy | Best constant baseline | Beats baseline? |
| --- | ---: | ---: | --- |
| counterfactual_control | 0.250000 | 0.333333 | False |
| hard_holdout | 0.416667 | 0.333333 | True |
| holdout | 0.416667 | 0.333333 | True |
| permuted_control | 0.277778 | 0.333333 | False |
| stress_holdout | 0.333333 | 0.333333 | False |
| train | 0.472222 | 0.333333 | True |

## Correlation summary

- accuracy_vs_active_count_ratio_mean: 0.178698
- accuracy_vs_branching_ratio_estimate: -0.074112
- accuracy_vs_neutral_occupancy: 0.074106

## Sample excerpt

| Mode | Seed | Split | Stimulus | Input vector | Branching | Neutral | Action | Expected | Outcome |
| --- | ---: | --- | --- | --- | ---: | ---: | --- | --- | --- |
| infer | 133 | train | low-drive-execute-train | [0.25, 0.0625, 0.0] | 0.357490 | 0.810855 | retry | execute | failure |
| infer | 133 | train | low-drive-retry-train | [-0.25, -0.0625, 0.0] | 0.187405 | 0.797697 | assertive | retry | failure |
| infer | 133 | train | low-drive-query-train | [0.0, 0.0, 0.0] | 0.000000 | 1.000000 | query | query | success |
| infer | 133 | holdout | low-drive-execute-holdout-noisy | [0.2, 0.05, 0.025] | 0.000000 | 1.000000 | query | execute | failure |
| infer | 133 | holdout | low-drive-retry-holdout-noisy | [-0.2, -0.05, 0.025] | 0.000000 | 1.000000 | query | retry | failure |
| infer | 133 | holdout | low-drive-query-holdout-low-drive | [0.0125, 0.0, -0.0125] | 0.000000 | 1.000000 | query | query | success |
| infer | 133 | hard_holdout | low-drive-execute-hard-low-positive | [0.0875, 0.0125, -0.025] | 0.000000 | 1.000000 | query | execute | failure |
| infer | 133 | hard_holdout | low-drive-execute-hard-conflict | [0.15, -0.1, 0.05] | 0.000000 | 1.000000 | query | execute | failure |
| infer | 133 | hard_holdout | low-drive-retry-hard-low-negative | [-0.0875, -0.0125, 0.05] | 0.000000 | 1.000000 | query | retry | failure |
| infer | 133 | hard_holdout | low-drive-retry-hard-conflict | [-0.15, 0.1, -0.05] | 0.000000 | 1.000000 | query | retry | failure |
| infer | 133 | hard_holdout | low-drive-query-hard-balanced | [0.025, -0.025, 0.0125] | 0.000000 | 1.000000 | query | query | success |
| infer | 133 | hard_holdout | low-drive-query-hard-weak-drive | [-0.0125, 0.0125, 0.025] | 0.000000 | 1.000000 | query | query | success |
| infer | 133 | permuted_control | low-drive-execute-train-permuted-as-retry | [0.25, 0.0625, 0.0] | 0.357490 | 0.810855 | retry | retry | success |
| infer | 133 | permuted_control | low-drive-retry-train-permuted-as-query | [-0.25, -0.0625, 0.0] | 0.187405 | 0.797697 | assertive | query | failure |
| infer | 133 | permuted_control | low-drive-query-train-permuted-as-execute | [0.0, 0.0, 0.0] | 0.000000 | 1.000000 | query | execute | failure |
| infer | 133 | stress_holdout | low-drive-execute-stress-low-margin | [0.045, -0.03, 0.02] | 0.000000 | 1.000000 | query | execute | failure |
| infer | 133 | stress_holdout | low-drive-execute-stress-near-neutral | [0.03, 0.02, -0.03] | 0.000000 | 1.000000 | query | execute | failure |
| infer | 133 | stress_holdout | low-drive-retry-stress-low-margin | [-0.045, 0.03, -0.02] | 0.000000 | 1.000000 | query | retry | failure |
| infer | 133 | stress_holdout | low-drive-retry-stress-near-neutral | [-0.03, -0.02, 0.03] | 0.000000 | 1.000000 | query | retry | failure |
| infer | 133 | stress_holdout | low-drive-query-stress-positive-negative | [0.045, -0.045, 0.0] | 0.000000 | 1.000000 | query | query | success |
| infer | 133 | stress_holdout | low-drive-query-stress-negative-positive | [-0.045, 0.045, 0.0] | 0.000000 | 1.000000 | query | query | success |
| infer | 133 | counterfactual_control | low-drive-execute-holdout-counterfactual-query | [0.2, 0.05, 0.025] | 0.000000 | 1.000000 | query | query | success |
| infer | 133 | counterfactual_control | low-drive-retry-holdout-counterfactual-execute | [-0.2, -0.05, 0.025] | 0.000000 | 1.000000 | query | execute | failure |
| infer | 133 | counterfactual_control | low-drive-query-holdout-counterfactual-retry | [0.0125, 0.0, -0.0125] | 0.000000 | 1.000000 | query | retry | failure |
| infer | 134 | train | low-drive-execute-train | [0.25, 0.0625, 0.0] | 0.305056 | 0.768092 | execute | execute | success |
| infer | 134 | train | low-drive-retry-train | [-0.25, -0.0625, 0.0] | 0.334728 | 0.741776 | query | retry | failure |
| infer | 134 | train | low-drive-query-train | [0.0, 0.0, 0.0] | 0.000000 | 1.000000 | query | query | success |
| infer | 134 | holdout | low-drive-execute-holdout-noisy | [0.2, 0.05, 0.025] | 0.000000 | 1.000000 | query | execute | failure |
| infer | 134 | holdout | low-drive-retry-holdout-noisy | [-0.2, -0.05, 0.025] | 0.000000 | 1.000000 | query | retry | failure |
| infer | 134 | holdout | low-drive-query-holdout-low-drive | [0.0125, 0.0, -0.0125] | 0.000000 | 1.000000 | query | query | success |
| infer | 134 | hard_holdout | low-drive-execute-hard-low-positive | [0.0875, 0.0125, -0.025] | 0.000000 | 1.000000 | query | execute | failure |
| infer | 134 | hard_holdout | low-drive-execute-hard-conflict | [0.15, -0.1, 0.05] | 0.000000 | 1.000000 | query | execute | failure |
| infer | 134 | hard_holdout | low-drive-retry-hard-low-negative | [-0.0875, -0.0125, 0.05] | 0.000000 | 1.000000 | query | retry | failure |
| infer | 134 | hard_holdout | low-drive-retry-hard-conflict | [-0.15, 0.1, -0.05] | 0.000000 | 1.000000 | query | retry | failure |
| infer | 134 | hard_holdout | low-drive-query-hard-balanced | [0.025, -0.025, 0.0125] | 0.000000 | 1.000000 | query | query | success |
| infer | 134 | hard_holdout | low-drive-query-hard-weak-drive | [-0.0125, 0.0125, 0.025] | 0.000000 | 1.000000 | query | query | success |
| infer | 134 | permuted_control | low-drive-execute-train-permuted-as-retry | [0.25, 0.0625, 0.0] | 0.305056 | 0.768092 | execute | retry | failure |
| infer | 134 | permuted_control | low-drive-retry-train-permuted-as-query | [-0.25, -0.0625, 0.0] | 0.334728 | 0.741776 | query | query | success |
| infer | 134 | permuted_control | low-drive-query-train-permuted-as-execute | [0.0, 0.0, 0.0] | 0.000000 | 1.000000 | query | execute | failure |
| infer | 134 | stress_holdout | low-drive-execute-stress-low-margin | [0.045, -0.03, 0.02] | 0.000000 | 1.000000 | query | execute | failure |
| infer | 134 | stress_holdout | low-drive-execute-stress-near-neutral | [0.03, 0.02, -0.03] | 0.000000 | 1.000000 | query | execute | failure |
| infer | 134 | stress_holdout | low-drive-retry-stress-low-margin | [-0.045, 0.03, -0.02] | 0.000000 | 1.000000 | query | retry | failure |
| infer | 134 | stress_holdout | low-drive-retry-stress-near-neutral | [-0.03, -0.02, 0.03] | 0.000000 | 1.000000 | query | retry | failure |
| infer | 134 | stress_holdout | low-drive-query-stress-positive-negative | [0.045, -0.045, 0.0] | 0.000000 | 1.000000 | query | query | success |
| infer | 134 | stress_holdout | low-drive-query-stress-negative-positive | [-0.045, 0.045, 0.0] | 0.000000 | 1.000000 | query | query | success |
| infer | 134 | counterfactual_control | low-drive-execute-holdout-counterfactual-query | [0.2, 0.05, 0.025] | 0.000000 | 1.000000 | query | query | success |
| infer | 134 | counterfactual_control | low-drive-retry-holdout-counterfactual-execute | [-0.2, -0.05, 0.025] | 0.000000 | 1.000000 | query | execute | failure |
| infer | 134 | counterfactual_control | low-drive-query-holdout-counterfactual-retry | [0.0125, 0.0, -0.0125] | 0.000000 | 1.000000 | query | retry | failure |
| train | 133 | train | low-drive-execute-train | [0.25, 0.0625, 0.0] | 0.062500 | 0.967105 | query | execute | failure |
| train | 133 | train | low-drive-retry-train | [-0.25, -0.0625, 0.0] | 0.062500 | 0.965461 | query | retry | failure |
| train | 133 | train | low-drive-query-train | [0.0, 0.0, 0.0] | 0.000000 | 1.000000 | query | query | success |
| train | 133 | holdout | low-drive-execute-holdout-noisy | [0.2, 0.05, 0.025] | 0.000000 | 1.000000 | query | execute | failure |
| train | 133 | holdout | low-drive-retry-holdout-noisy | [-0.2, -0.05, 0.025] | 0.000000 | 1.000000 | query | retry | failure |
| train | 133 | holdout | low-drive-query-holdout-low-drive | [0.0125, 0.0, -0.0125] | 0.000000 | 1.000000 | query | query | success |
| train | 133 | hard_holdout | low-drive-execute-hard-low-positive | [0.0875, 0.0125, -0.025] | 0.000000 | 1.000000 | query | execute | failure |
| train | 133 | hard_holdout | low-drive-execute-hard-conflict | [0.15, -0.1, 0.05] | 0.000000 | 1.000000 | query | execute | failure |
| train | 133 | hard_holdout | low-drive-retry-hard-low-negative | [-0.0875, -0.0125, 0.05] | 0.000000 | 1.000000 | query | retry | failure |
| train | 133 | hard_holdout | low-drive-retry-hard-conflict | [-0.15, 0.1, -0.05] | 0.000000 | 1.000000 | query | retry | failure |
| train | 133 | hard_holdout | low-drive-query-hard-balanced | [0.025, -0.025, 0.0125] | 0.000000 | 1.000000 | query | query | success |
| train | 133 | hard_holdout | low-drive-query-hard-weak-drive | [-0.0125, 0.0125, 0.025] | 0.000000 | 1.000000 | query | query | success |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | 228 more samples |

## Verdict

Controlled-regime estimator calibration moved/recorded branching and occupancy metrics across drive settings (branching range 0.000000..0.357490), but no drive regime beat the best constant baseline on stress_holdout; criticality remains diagnostic instrumentation, not intelligence evidence.

## Evidence boundary

This is controlled-regime estimator calibration for Qubic NIA Vol. 8 style criticality diagnostics; drive-dependent branching or occupancy movement is not intelligence evidence unless stress/control task quality beats constant baselines.

## Notes

- uses low/medium/high input-drive scales over the same held-out/stress/control cases
- stress_holdout and control labels remain evaluation-only; no fitness callback is used
- reports estimator movement, action distributions and split baselines separately
