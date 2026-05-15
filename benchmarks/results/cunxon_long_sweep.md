# cuNxon long sweep action diagnostic

Status: `long-sweep probe viable`
Samples: 108

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Step horizons: 32, 512, 4096, 32768
- Seed offsets: 79, 80, 81

## Why this probe exists

Earlier cuNxon probes showed flat or baseline-level action readouts. This long sweep keeps the task-coupled action contract but tests longer horizons, multiple seeds, frozen inference, paper-canonical train mode, and train mode with simple reward/neuromodulator injection before treating the backend as useful.

## Long sweep summary

| Mode | Steps | Accuracy | Unique readouts | Action distribution |
| --- | ---: | ---: | ---: | --- |
| infer | 32 | 0.333333 | 3 | query=8, retry=1 |
| infer | 512 | 0.333333 | 2 | execute=1, query=8 |
| infer | 4096 | 0.333333 | 1 | query=9 |
| infer | 32768 | 0.333333 | 1 | query=9 |
| train | 32 | 0.333333 | 1 | query=9 |
| train | 512 | 0.333333 | 1 | query=9 |
| train | 4096 | 0.333333 | 1 | query=9 |
| train | 32768 | 0.333333 | 1 | query=9 |
| train_rewarded | 32 | 0.333333 | 1 | query=9 |
| train_rewarded | 512 | 0.333333 | 1 | query=9 |
| train_rewarded | 4096 | 0.333333 | 1 | query=9 |
| train_rewarded | 32768 | 0.333333 | 1 | query=9 |

## Trivial baselines

| Baseline | Accuracy |
| --- | ---: |
| always_execute | 0.333333 |
| always_query | 0.333333 |
| always_retry | 0.333333 |

## Samples

| Mode | Steps | Seed | Stimulus | Expected | Readout | Decoded | Outcome | Energy |
| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| infer | 32 | 79 | execute-positive-drive | execute | [0, 0, -1] | RETRY (retry, 0.3333) | failure | 708.582 |
| infer | 32 | 79 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 719.918 |
| infer | 32 | 79 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| infer | 32 | 80 | execute-positive-drive | execute | [+1, -1, 0] | PAUSE (query, 0.3333) | failure | 583.819 |
| infer | 32 | 80 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 631.858 |
| infer | 32 | 80 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| infer | 32 | 81 | execute-positive-drive | execute | [+1, -1, 0] | PAUSE (query, 0.3333) | failure | 626.732 |
| infer | 32 | 81 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 635.594 |
| infer | 32 | 81 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| infer | 512 | 79 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 8702.66 |
| infer | 512 | 79 | retry-negative-drive | retry | [0, +1, 0] | PROCEED (execute, 0.3333) | failure | 8572.69 |
| infer | 512 | 79 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| infer | 512 | 80 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 6900.42 |
| infer | 512 | 80 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 8269.75 |
| infer | 512 | 80 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| infer | 512 | 81 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 8044.68 |
| infer | 512 | 81 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 7521.6 |
| infer | 512 | 81 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| infer | 4096 | 79 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 68634.8 |
| infer | 4096 | 79 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 66716.1 |
| infer | 4096 | 79 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| infer | 4096 | 80 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 54577.7 |
| infer | 4096 | 80 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 65367.1 |
| infer | 4096 | 80 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| infer | 4096 | 81 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 63316 |
| infer | 4096 | 81 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 58710.6 |
| infer | 4096 | 81 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| infer | 32768 | 79 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 554400 |
| infer | 32768 | 79 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 534266 |
| infer | 32768 | 79 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| infer | 32768 | 80 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 437458 |
| infer | 32768 | 80 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 498707 |
| infer | 32768 | 80 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| infer | 32768 | 81 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 509390 |
| infer | 32768 | 81 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 470743 |
| infer | 32768 | 81 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train | 32 | 79 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 46.218 |
| train | 32 | 79 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 47.4392 |
| train | 32 | 79 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train | 32 | 80 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 24.5663 |
| train | 32 | 80 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 23.0728 |
| train | 32 | 80 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train | 32 | 81 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 29.6949 |
| train | 32 | 81 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 30.1628 |
| train | 32 | 81 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train | 512 | 79 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 54.422 |
| train | 512 | 79 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 55.6172 |
| train | 512 | 79 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train | 512 | 80 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 29.5827 |
| train | 512 | 80 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 27.7222 |
| train | 512 | 80 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train | 512 | 81 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 35.0515 |
| train | 512 | 81 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 35.4512 |
| train | 512 | 81 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train | 4096 | 79 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 54.422 |
| train | 4096 | 79 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 55.6172 |
| train | 4096 | 79 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train | 4096 | 80 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 29.5827 |
| train | 4096 | 80 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 27.7222 |
| train | 4096 | 80 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train | 4096 | 81 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 35.0515 |
| train | 4096 | 81 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 35.4512 |
| train | 4096 | 81 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train | 32768 | 79 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 54.422 |
| train | 32768 | 79 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 55.6172 |
| train | 32768 | 79 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train | 32768 | 80 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 29.5827 |
| train | 32768 | 80 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 27.7222 |
| train | 32768 | 80 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train | 32768 | 81 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 35.0515 |
| train | 32768 | 81 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 35.4512 |
| train | 32768 | 81 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train_rewarded | 32 | 79 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 45.9719 |
| train_rewarded | 32 | 79 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 50.3177 |
| train_rewarded | 32 | 79 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train_rewarded | 32 | 80 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 24.2377 |
| train_rewarded | 32 | 80 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 26.3262 |
| train_rewarded | 32 | 80 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train_rewarded | 32 | 81 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 29.6081 |
| train_rewarded | 32 | 81 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 33.4253 |
| train_rewarded | 32 | 81 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train_rewarded | 512 | 79 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 54.1452 |
| train_rewarded | 512 | 79 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 58.9338 |
| train_rewarded | 512 | 79 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train_rewarded | 512 | 80 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 29.1833 |
| train_rewarded | 512 | 80 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 31.3625 |
| train_rewarded | 512 | 80 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train_rewarded | 512 | 81 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 34.9396 |
| train_rewarded | 512 | 81 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 39.207 |
| train_rewarded | 512 | 81 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train_rewarded | 4096 | 79 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 54.1452 |
| train_rewarded | 4096 | 79 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 58.9338 |
| train_rewarded | 4096 | 79 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train_rewarded | 4096 | 80 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 29.1833 |
| train_rewarded | 4096 | 80 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 31.3625 |
| train_rewarded | 4096 | 80 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train_rewarded | 4096 | 81 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 34.9396 |
| train_rewarded | 4096 | 81 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 39.207 |
| train_rewarded | 4096 | 81 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train_rewarded | 32768 | 79 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 54.1452 |
| train_rewarded | 32768 | 79 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 58.9338 |
| train_rewarded | 32768 | 79 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train_rewarded | 32768 | 80 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 29.1833 |
| train_rewarded | 32768 | 80 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 31.3625 |
| train_rewarded | 32768 | 80 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |
| train_rewarded | 32768 | 81 | execute-positive-drive | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 34.9396 |
| train_rewarded | 32768 | 81 | retry-negative-drive | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 39.207 |
| train_rewarded | 32768 | 81 | query-neutral-drive | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 |

## Notes

- fresh one-sphere network per mode/steps/seed/stimulus sample
- infer is frozen; train is plastic; train_rewarded adds simple neuromodulator feedback
- longer horizons are diagnostic evidence, not benchmark success by themselves

## Evidence boundary

Longer horizons, reward injection, readout diversity, or one successful sample does not prove intelligence, generalization, or useful learning. This diagnostic becomes positive evidence only if it beats trivial baselines across seeds and horizons.
