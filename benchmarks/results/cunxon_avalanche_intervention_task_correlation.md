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
- Seed offsets: 127, 128
- Samples: 192

## Why this probe exists

This report follows Qubic NIA Vol. 8's branching-ratio/criticality claims by checking whether bounded avalanche/regime movements are coupled to held-out, stress and control action quality. The relevant comparator is constant baselines; criticality metrics alone are not intelligence evidence.

## Intervention configurations

| Config | Steps | Interval | Samples | Mean branching | Branching range | Mean neutral | Splits beating constants |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| short-dense-heldout-stress | 128 | 8 | 96 | 0.100726 | 0.000000..0.447631 | 0.924770 | hard_holdout, holdout |
| baseline-equivalent-heldout-stress | 256 | 16 | 96 | 0.120770 | 0.000000..0.501736 | 0.944970 | hard_holdout, holdout |

## Split accuracy versus constant baselines

| Split | Accuracy | Best constant baseline | Beats baseline? |
| --- | ---: | ---: | --- |
| counterfactual_control | 0.250000 | 0.333333 | False |
| hard_holdout | 0.458333 | 0.333333 | True |
| holdout | 0.458333 | 0.333333 | True |
| permuted_control | 0.208333 | 0.333333 | False |
| stress_holdout | 0.333333 | 0.333333 | False |
| train | 0.333333 | 0.333333 | False |

## Correlation summary

- accuracy_vs_active_count_ratio_mean: 0.044276
- accuracy_vs_branching_ratio_estimate: -0.176464
- accuracy_vs_neutral_occupancy: 0.183790

## Sample excerpt

| Elapsed | Mode | Seed | Split | Stimulus | Branching | Neutral | Action | Expected | Outcome |
| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- | --- |
| 8ms | infer | 127 | train | execute-train | 0.257755 | 0.713816 | query | execute | failure |
| 7ms | infer | 127 | train | retry-train | 0.272400 | 0.652961 | assertive | retry | failure |
| 7ms | infer | 127 | train | query-train | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 127 | holdout | execute-holdout-noisy | 0.340585 | 0.768092 | execute | execute | success |
| 7ms | infer | 127 | holdout | retry-holdout-noisy | 0.275521 | 0.769737 | query | retry | failure |
| 7ms | infer | 127 | holdout | query-holdout-low-drive | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 127 | hard_holdout | execute-hard-low-positive | 0.340585 | 0.768092 | execute | execute | success |
| 7ms | infer | 127 | hard_holdout | execute-hard-conflict | 0.265562 | 0.667763 | query | execute | failure |
| 7ms | infer | 127 | hard_holdout | retry-hard-low-negative | 0.275521 | 0.769737 | query | retry | failure |
| 7ms | infer | 127 | hard_holdout | retry-hard-conflict | 0.343750 | 0.796053 | query | retry | failure |
| 7ms | infer | 127 | hard_holdout | query-hard-balanced | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 127 | hard_holdout | query-hard-weak-drive | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 127 | permuted_control | execute-train-permuted-as-retry | 0.257755 | 0.713816 | query | retry | failure |
| 7ms | infer | 127 | permuted_control | retry-train-permuted-as-query | 0.272400 | 0.652961 | assertive | query | failure |
| 8ms | infer | 127 | permuted_control | query-train-permuted-as-execute | 0.000000 | 1.000000 | query | execute | failure |
| 8ms | infer | 127 | stress_holdout | execute-stress-low-margin | 0.000000 | 1.000000 | query | execute | failure |
| 8ms | infer | 127 | stress_holdout | execute-stress-near-neutral | 0.000000 | 1.000000 | query | execute | failure |
| 8ms | infer | 127 | stress_holdout | retry-stress-low-margin | 0.000000 | 1.000000 | query | retry | failure |
| 8ms | infer | 127 | stress_holdout | retry-stress-near-neutral | 0.000000 | 1.000000 | query | retry | failure |
| 8ms | infer | 127 | stress_holdout | query-stress-positive-negative | 0.000000 | 1.000000 | query | query | success |
| 8ms | infer | 127 | stress_holdout | query-stress-negative-positive | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 127 | counterfactual_control | execute-holdout-counterfactual-query | 0.340585 | 0.768092 | execute | query | failure |
| 7ms | infer | 127 | counterfactual_control | retry-holdout-counterfactual-execute | 0.275521 | 0.769737 | query | execute | failure |
| 7ms | infer | 127 | counterfactual_control | query-holdout-counterfactual-retry | 0.000000 | 1.000000 | query | retry | failure |
| 7ms | infer | 128 | train | execute-train | 0.325984 | 0.740132 | query | execute | failure |
| 7ms | infer | 128 | train | retry-train | 0.302189 | 0.708882 | query | retry | failure |
| 7ms | infer | 128 | train | query-train | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 128 | holdout | execute-holdout-noisy | 0.325789 | 0.786184 | execute | execute | success |
| 7ms | infer | 128 | holdout | retry-holdout-noisy | 0.447631 | 0.843750 | query | retry | failure |
| 8ms | infer | 128 | holdout | query-holdout-low-drive | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 128 | hard_holdout | execute-hard-low-positive | 0.325789 | 0.786184 | execute | execute | success |
| 7ms | infer | 128 | hard_holdout | execute-hard-conflict | 0.322892 | 0.759868 | execute | execute | success |
| 7ms | infer | 128 | hard_holdout | retry-hard-low-negative | 0.447631 | 0.843750 | query | retry | failure |
| 7ms | infer | 128 | hard_holdout | retry-hard-conflict | 0.322509 | 0.761513 | query | retry | failure |
| 7ms | infer | 128 | hard_holdout | query-hard-balanced | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 128 | hard_holdout | query-hard-weak-drive | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 128 | permuted_control | execute-train-permuted-as-retry | 0.325984 | 0.740132 | query | retry | failure |
| 7ms | infer | 128 | permuted_control | retry-train-permuted-as-query | 0.302189 | 0.708882 | query | query | success |
| 7ms | infer | 128 | permuted_control | query-train-permuted-as-execute | 0.000000 | 1.000000 | query | execute | failure |
| 7ms | infer | 128 | stress_holdout | execute-stress-low-margin | 0.000000 | 1.000000 | query | execute | failure |
| 7ms | infer | 128 | stress_holdout | execute-stress-near-neutral | 0.000000 | 1.000000 | query | execute | failure |
| 7ms | infer | 128 | stress_holdout | retry-stress-low-margin | 0.000000 | 1.000000 | query | retry | failure |
| 7ms | infer | 128 | stress_holdout | retry-stress-near-neutral | 0.000000 | 1.000000 | query | retry | failure |
| 7ms | infer | 128 | stress_holdout | query-stress-positive-negative | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 128 | stress_holdout | query-stress-negative-positive | 0.000000 | 1.000000 | query | query | success |
| 7ms | infer | 128 | counterfactual_control | execute-holdout-counterfactual-query | 0.325789 | 0.786184 | execute | query | failure |
| 7ms | infer | 128 | counterfactual_control | retry-holdout-counterfactual-execute | 0.447631 | 0.843750 | query | execute | failure |
| 7ms | infer | 128 | counterfactual_control | query-holdout-counterfactual-retry | 0.000000 | 1.000000 | query | retry | failure |
| 9ms | train | 127 | train | execute-train | 0.083333 | 0.932566 | query | execute | failure |
| 9ms | train | 127 | train | retry-train | 0.116667 | 0.914474 | query | retry | failure |
| 9ms | train | 127 | train | query-train | 0.000000 | 1.000000 | query | query | success |
| 9ms | train | 127 | holdout | execute-holdout-noisy | 0.062500 | 0.960526 | query | execute | failure |
| 9ms | train | 127 | holdout | retry-holdout-noisy | 0.062500 | 0.955592 | query | retry | failure |
| 9ms | train | 127 | holdout | query-holdout-low-drive | 0.000000 | 1.000000 | query | query | success |
| 9ms | train | 127 | hard_holdout | execute-hard-low-positive | 0.062500 | 0.960526 | query | execute | failure |
| 9ms | train | 127 | hard_holdout | execute-hard-conflict | 0.062500 | 0.935855 | query | execute | failure |
| 9ms | train | 127 | hard_holdout | retry-hard-low-negative | 0.062500 | 0.955592 | query | retry | failure |
| 9ms | train | 127 | hard_holdout | retry-hard-conflict | 0.083333 | 0.927632 | query | retry | failure |
| 9ms | train | 127 | hard_holdout | query-hard-balanced | 0.000000 | 1.000000 | query | query | success |
| 9ms | train | 127 | hard_holdout | query-hard-weak-drive | 0.000000 | 1.000000 | query | query | success |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | 132 more samples |

## Verdict

Bounded task-coupled avalanche interventions moved/recorded regime metrics (mean branching estimate 0.110748), but no configuration beat the best constant baseline on stress_holdout; criticality remains diagnostic instrumentation, not intelligence evidence.

## Evidence boundary

This is task-coupled regime instrumentation from bounded cuNxon snapshot windows; avalanche movement or branching-ratio estimates are not intelligence evidence unless held-out/stress task quality beats constant baselines under controls.

## Notes

- uses fresh seeds after the estimator-sensitivity matrix
- scores train, holdout, stress_holdout, counterfactual_control and permuted_control splits
- constant-action baselines are computed per split before interpreting regime movement
