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
- Seed offsets: 125, 126
- Steps per window: 256
- Snapshot sample interval: 16

## Why this probe exists

Qubic NIA Vol. 8 frames branching ratio as an operational criticality indicator. This probe uses full-sphere snapshot windows rather than only final action readouts, then compares the branching-ratio estimate with action-contract accuracy and trivial baselines.

## Snapshot window metrics

| Mode | Seed | Stimulus | Branching-ratio estimate | Active-ratio mean | Neutral occupancy | Avalanche events | Mean/max avalanche | Final action | Outcome |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| infer | 125 | execute-positive-drive | 0.363658 | 0.937018 | 0.754934 | 14 | 4.125000/29 | query | failure |
| infer | 125 | retry-negative-drive | 0.426655 | 1.052044 | 0.787829 | 15 | 3.812500/21 | query | failure |
| infer | 125 | query-neutral-drive | 0.000000 | 1.000000 | 1.000000 | 0 | 0.000000/0 | query | success |
| infer | 126 | execute-positive-drive | 0.358624 | 0.974199 | 0.743421 | 14 | 4.187500/29 | query | failure |
| infer | 126 | retry-negative-drive | 0.382948 | 0.952742 | 0.761513 | 16 | 4.187500/28 | execute | failure |
| infer | 126 | query-neutral-drive | 0.000000 | 1.000000 | 1.000000 | 0 | 0.000000/0 | query | success |
| train | 125 | execute-positive-drive | 0.104167 | 0.988889 | 0.940789 | 3 | 0.375000/4 | query | failure |
| train | 125 | retry-negative-drive | 0.104167 | 0.983333 | 0.939145 | 3 | 0.375000/4 | query | failure |
| train | 125 | query-neutral-drive | 0.000000 | 1.000000 | 1.000000 | 0 | 0.000000/0 | query | success |
| train | 126 | execute-positive-drive | 0.062500 | 0.966667 | 0.944079 | 1 | 0.250000/4 | query | failure |
| train | 126 | retry-negative-drive | 0.083333 | 0.971111 | 0.940789 | 2 | 0.375000/5 | query | failure |
| train | 126 | query-neutral-drive | 0.000000 | 1.000000 | 1.000000 | 0 | 0.000000/0 | query | success |

## Accuracy by mode

- infer: 0.333333
- train: 0.333333

## Trivial baselines

- always_execute: 0.333333
- always_query: 0.333333
- always_retry: 0.333333

## Correlation summary

- accuracy_vs_active_count_ratio_mean: 0.366361
- accuracy_vs_branching_ratio_estimate: -0.674629
- accuracy_vs_neutral_occupancy: 0.688052

## Verdict

Snapshot-level avalanche metrics were captured (mean branching-ratio estimate 0.157171), but action quality did not beat the best constant baseline for every mode; criticality remains instrumentation, not sufficient evidence.

## Notes

- captures full-sphere state snapshots at bounded step intervals
- branching-ratio estimate uses new activations divided by previous active states
- action score uses the existing project action contract and trivial baselines
- snapshot criticality diagnostics are not intelligence evidence by themselves

## Evidence boundary

This is a full-sphere snapshot-level branching/avalanche diagnostic. A branching-ratio estimate or visible avalanche activity is not intelligence evidence unless task quality beats baselines on held-out/control cases.
