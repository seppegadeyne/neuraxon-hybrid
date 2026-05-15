# cuNxon avalanche-window snapshot probe

Status: `avalanche-window probe completed`

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Modes: infer, train
- Seed offsets: 122, 123, 124
- Steps per window: 256
- Snapshot sample interval: 16

## Why this probe exists

Qubic NIA Vol. 8 frames branching ratio as an operational criticality indicator. This probe uses full-sphere snapshot windows rather than only final action readouts, then compares the branching-ratio estimate with action-contract accuracy and trivial baselines.

## Snapshot window metrics

| Mode | Seed | Stimulus | Branching-ratio estimate | Active-ratio mean | Neutral occupancy | Avalanche events | Mean/max avalanche | Final action | Outcome |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| infer | 122 | execute-positive-drive | 0.284913 | 0.975147 | 0.756579 | 15 | 3.437500/25 | assertive | failure |
| infer | 122 | retry-negative-drive | 0.444301 | 1.007760 | 0.796053 | 16 | 4.000000/27 | execute | failure |
| infer | 122 | query-neutral-drive | 0.000000 | 1.000000 | 1.000000 | 0 | 0.000000/0 | query | success |
| infer | 123 | execute-positive-drive | 0.427827 | 0.968175 | 0.805921 | 15 | 3.875000/28 | query | failure |
| infer | 123 | retry-negative-drive | 0.335372 | 0.940496 | 0.758224 | 13 | 3.812500/26 | query | failure |
| infer | 123 | query-neutral-drive | 0.000000 | 1.000000 | 1.000000 | 0 | 0.000000/0 | query | success |
| infer | 124 | execute-positive-drive | 0.389367 | 1.025310 | 0.781250 | 15 | 4.000000/30 | query | failure |
| infer | 124 | retry-negative-drive | 0.378322 | 1.054628 | 0.791118 | 12 | 3.937500/26 | query | failure |
| infer | 124 | query-neutral-drive | 0.000000 | 1.000000 | 1.000000 | 0 | 0.000000/0 | query | success |
| train | 122 | execute-positive-drive | 0.083333 | 0.977778 | 0.942434 | 2 | 0.312500/4 | query | failure |
| train | 122 | retry-negative-drive | 0.125000 | 0.977778 | 0.935855 | 4 | 0.562500/6 | query | failure |
| train | 122 | query-neutral-drive | 0.000000 | 1.000000 | 1.000000 | 0 | 0.000000/0 | query | success |
| train | 123 | execute-positive-drive | 0.062500 | 1.000000 | 0.947368 | 1 | 0.125000/2 | query | failure |
| train | 123 | retry-negative-drive | 0.083333 | 0.971111 | 0.940789 | 2 | 0.375000/5 | query | failure |
| train | 123 | query-neutral-drive | 0.000000 | 1.000000 | 1.000000 | 0 | 0.000000/0 | query | success |
| train | 124 | execute-positive-drive | 0.062500 | 1.000000 | 0.947368 | 1 | 0.125000/2 | query | failure |
| train | 124 | retry-negative-drive | 0.104167 | 1.000000 | 0.942434 | 3 | 0.312500/3 | query | failure |
| train | 124 | query-neutral-drive | 0.000000 | 1.000000 | 1.000000 | 0 | 0.000000/0 | query | success |

## Accuracy by mode

- infer: 0.333333
- train: 0.333333

## Trivial baselines

- always_execute: 0.333333
- always_query: 0.333333
- always_retry: 0.333333

## Correlation summary

- accuracy_vs_active_count_ratio_mean: 0.168834
- accuracy_vs_branching_ratio_estimate: -0.663921
- accuracy_vs_neutral_occupancy: 0.697874

## Verdict

Snapshot-level avalanche metrics were captured (mean branching-ratio estimate 0.154496), but action quality did not beat the best constant baseline for every mode; criticality remains instrumentation, not sufficient evidence.

## Notes

- captures full-sphere state snapshots at bounded step intervals
- branching-ratio estimate uses new activations divided by previous active states
- action score uses the existing project action contract and trivial baselines
- snapshot criticality diagnostics are not intelligence evidence by themselves

## Evidence boundary

This is a full-sphere snapshot-level branching/avalanche diagnostic. A branching-ratio estimate or visible avalanche activity is not intelligence evidence unless task quality beats baselines on held-out/control cases.
