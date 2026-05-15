# cuNxon avalanche intervention/task correlation

Status: `avalanche intervention/task correlation completed`
Hypothesis: `avalanche_intervention_task_correlation`

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Modes: infer, train
- Seed offsets: 129, 130, 131, 132
- Samples: 384

## Why this probe exists

This report follows Qubic NIA Vol. 8's branching-ratio/criticality claims by checking whether bounded avalanche/regime movements are coupled to held-out, stress and control action quality. The relevant comparator is constant baselines; criticality metrics alone are not intelligence evidence.

## Intervention configurations

| Config | Steps | Interval | Samples | Mean branching | Branching range | Mean neutral | Splits beating constants |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| short-dense-heldout-stress | 128 | 8 | 192 | 0.099172 | 0.000000..0.496279 | 0.917960 | hard_holdout, holdout, train |
| baseline-equivalent-heldout-stress | 256 | 16 | 192 | 0.125668 | 0.000000..0.565104 | 0.939693 | hard_holdout, holdout, train |

## Split accuracy versus constant baselines

| Split | Accuracy | Best constant baseline | Beats baseline? |
| --- | ---: | ---: | --- |
| counterfactual_control | 0.333333 | 0.333333 | False |
| hard_holdout | 0.447917 | 0.333333 | True |
| holdout | 0.437500 | 0.333333 | True |
| permuted_control | 0.250000 | 0.333333 | False |
| stress_holdout | 0.333333 | 0.333333 | False |
| train | 0.437500 | 0.333333 | True |

## Correlation summary

- accuracy_vs_active_count_ratio_mean: 0.139883
- accuracy_vs_branching_ratio_estimate: -0.082062
- accuracy_vs_neutral_occupancy: 0.118083

## Sample excerpt

| Elapsed | Mode | Seed | Split | Stimulus | Branching | Neutral | Action | Expected | Outcome |
| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- | --- |
| 8ms | infer | 129 | train | execute-train | 0.308110 | 0.710526 | execute | execute | success |
| 7ms | infer | 129 | train | retry-train | 0.252639 | 0.661184 | execute | retry | failure |
| 7ms | infer | 129 | train | query-train | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 129 | holdout | execute-holdout-noisy | 0.279407 | 0.753289 | query | execute | failure |
| 7ms | infer | 129 | holdout | retry-holdout-noisy | 0.347867 | 0.740132 | execute | retry | failure |
| 7ms | infer | 129 | holdout | query-holdout-low-drive | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 129 | hard_holdout | execute-hard-low-positive | 0.279407 | 0.753289 | query | execute | failure |
| 7ms | infer | 129 | hard_holdout | execute-hard-conflict | 0.270398 | 0.690789 | query | execute | failure |
| 7ms | infer | 129 | hard_holdout | retry-hard-low-negative | 0.347867 | 0.740132 | execute | retry | failure |
| 8ms | infer | 129 | hard_holdout | retry-hard-conflict | 0.297248 | 0.682566 | execute | retry | failure |
| 8ms | infer | 129 | hard_holdout | query-hard-balanced | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 129 | hard_holdout | query-hard-weak-drive | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 129 | permuted_control | execute-train-permuted-as-retry | 0.308110 | 0.710526 | execute | retry | failure |
| 7ms | infer | 129 | permuted_control | retry-train-permuted-as-query | 0.252639 | 0.661184 | execute | query | failure |
| 7ms | infer | 129 | permuted_control | query-train-permuted-as-execute | 0.000000 | 1.000000 | query | execute | failure |
| 7ms | infer | 129 | stress_holdout | execute-stress-low-margin | 0.000000 | 1.000000 | query | execute | failure |
| 7ms | infer | 129 | stress_holdout | execute-stress-near-neutral | 0.000000 | 1.000000 | query | execute | failure |
| 7ms | infer | 129 | stress_holdout | retry-stress-low-margin | 0.000000 | 1.000000 | query | retry | failure |
| 7ms | infer | 129 | stress_holdout | retry-stress-near-neutral | 0.000000 | 1.000000 | query | retry | failure |
| 7ms | infer | 129 | stress_holdout | query-stress-positive-negative | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 129 | stress_holdout | query-stress-negative-positive | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 129 | counterfactual_control | execute-holdout-counterfactual-query | 0.279407 | 0.753289 | query | query | success |
| 8ms | infer | 129 | counterfactual_control | retry-holdout-counterfactual-execute | 0.347867 | 0.740132 | execute | execute | success |
| 8ms | infer | 129 | counterfactual_control | query-holdout-counterfactual-retry | 0.000000 | 1.000000 | query | retry | failure |
| 8ms | infer | 130 | train | execute-train | 0.344271 | 0.766447 | execute | execute | success |
| 8ms | infer | 130 | train | retry-train | 0.294638 | 0.728618 | execute | retry | failure |
| 8ms | infer | 130 | train | query-train | 0.000000 | 1.000000 | query | query | success |
| 8ms | infer | 130 | holdout | execute-holdout-noisy | 0.354814 | 0.789474 | execute | execute | success |
| 8ms | infer | 130 | holdout | retry-holdout-noisy | 0.496279 | 0.809211 | retry | retry | success |
| 8ms | infer | 130 | holdout | query-holdout-low-drive | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 130 | hard_holdout | execute-hard-low-positive | 0.354814 | 0.789474 | execute | execute | success |
| 7ms | infer | 130 | hard_holdout | execute-hard-conflict | 0.322512 | 0.699013 | assertive | execute | failure |
| 7ms | infer | 130 | hard_holdout | retry-hard-low-negative | 0.496279 | 0.809211 | retry | retry | success |
| 7ms | infer | 130 | hard_holdout | retry-hard-conflict | 0.292004 | 0.761513 | query | retry | failure |
| 7ms | infer | 130 | hard_holdout | query-hard-balanced | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 130 | hard_holdout | query-hard-weak-drive | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 130 | permuted_control | execute-train-permuted-as-retry | 0.344271 | 0.766447 | execute | retry | failure |
| 7ms | infer | 130 | permuted_control | retry-train-permuted-as-query | 0.294638 | 0.728618 | execute | query | failure |
| 7ms | infer | 130 | permuted_control | query-train-permuted-as-execute | 0.000000 | 1.000000 | query | execute | failure |
| 7ms | infer | 130 | stress_holdout | execute-stress-low-margin | 0.000000 | 1.000000 | query | execute | failure |
| 7ms | infer | 130 | stress_holdout | execute-stress-near-neutral | 0.000000 | 1.000000 | query | execute | failure |
| 7ms | infer | 130 | stress_holdout | retry-stress-low-margin | 0.000000 | 1.000000 | query | retry | failure |
| 7ms | infer | 130 | stress_holdout | retry-stress-near-neutral | 0.000000 | 1.000000 | query | retry | failure |
| 7ms | infer | 130 | stress_holdout | query-stress-positive-negative | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 130 | stress_holdout | query-stress-negative-positive | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 130 | counterfactual_control | execute-holdout-counterfactual-query | 0.354814 | 0.789474 | execute | query | failure |
| 7ms | infer | 130 | counterfactual_control | retry-holdout-counterfactual-execute | 0.496279 | 0.809211 | retry | execute | failure |
| 7ms | infer | 130 | counterfactual_control | query-holdout-counterfactual-retry | 0.000000 | 1.000000 | query | retry | failure |
| 7ms | infer | 131 | train | execute-train | 0.346396 | 0.756579 | query | execute | failure |
| 7ms | infer | 131 | train | retry-train | 0.317189 | 0.695724 | query | retry | failure |
| 7ms | infer | 131 | train | query-train | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 131 | holdout | execute-holdout-noisy | 0.411130 | 0.763158 | query | execute | failure |
| 7ms | infer | 131 | holdout | retry-holdout-noisy | 0.289732 | 0.753289 | execute | retry | failure |
| 7ms | infer | 131 | holdout | query-holdout-low-drive | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 131 | hard_holdout | execute-hard-low-positive | 0.411130 | 0.763158 | query | execute | failure |
| 7ms | infer | 131 | hard_holdout | execute-hard-conflict | 0.213723 | 0.717105 | execute | execute | success |
| 7ms | infer | 131 | hard_holdout | retry-hard-low-negative | 0.289732 | 0.753289 | execute | retry | failure |
| 7ms | infer | 131 | hard_holdout | retry-hard-conflict | 0.327491 | 0.725329 | retry | retry | success |
| 7ms | infer | 131 | hard_holdout | query-hard-balanced | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 131 | hard_holdout | query-hard-weak-drive | 0.000000 | 1.000000 | query | query | success |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | 324 more samples |

## Verdict

Bounded task-coupled avalanche interventions moved/recorded regime metrics (mean branching estimate 0.112420), but no configuration beat the best constant baseline on stress_holdout; criticality remains diagnostic instrumentation, not intelligence evidence.

## Evidence boundary

This is task-coupled regime instrumentation from bounded cuNxon snapshot windows; avalanche movement or branching-ratio estimates are not intelligence evidence unless held-out/stress task quality beats constant baselines under controls.

## Notes

- uses fresh seeds after the estimator-sensitivity matrix
- scores train, holdout, stress_holdout, counterfactual_control and permuted_control splits
- constant-action baselines are computed per split before interpreting regime movement
