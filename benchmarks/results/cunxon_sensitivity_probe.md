# cuNxon infer-vs-train sensitivity probe

Status: `sensitivity probe viable`
Samples: 18

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Steps per sample: 32

## Why this probe exists

The previous cuNxon action probe used frozen `cunxonNetworkStepInfer`. Upstream documents `cunxonNetworkStepTrain` as the paper-canonical continuous-learning mode, so this diagnostic compares frozen infer readouts with plastic train readouts on the same stimuli/seeds before claiming a richer adapter works.

## Mode summary

| Mode | Unique readouts | Action distribution |
| --- | ---: | --- |
| infer | 3 | query=8, retry=1 |
| train | 1 | query=9 |

## Stimulus sensitivity

| Stimulus | action changes by stimulus |
| --- | ---: |
| execute-positive-drive | 1 |
| query-neutral-drive | 0 |
| retry-negative-drive | 0 |

## Samples

| Mode | Seed | Stimulus | Input | Readout | Decoded | Normalized | Energy |
| --- | ---: | --- | --- | --- | --- | --- | ---: |
| infer | 79 | execute-positive-drive | [1, 0.25, 0] | [0, 0, -1] | RETRY (0.3333) | retry | 708.582 |
| infer | 79 | retry-negative-drive | [-1, -0.25, 0] | [0, 0, 0] | PAUSE (1.0000) | query | 719.918 |
| infer | 79 | query-neutral-drive | [0, 0, 0] | [0, 0, 0] | PAUSE (1.0000) | query | 0 |
| infer | 80 | execute-positive-drive | [1, 0.25, 0] | [+1, -1, 0] | PAUSE (0.3333) | query | 583.819 |
| infer | 80 | retry-negative-drive | [-1, -0.25, 0] | [0, 0, 0] | PAUSE (1.0000) | query | 631.858 |
| infer | 80 | query-neutral-drive | [0, 0, 0] | [0, 0, 0] | PAUSE (1.0000) | query | 0 |
| infer | 81 | execute-positive-drive | [1, 0.25, 0] | [+1, -1, 0] | PAUSE (0.3333) | query | 626.732 |
| infer | 81 | retry-negative-drive | [-1, -0.25, 0] | [0, 0, 0] | PAUSE (1.0000) | query | 635.594 |
| infer | 81 | query-neutral-drive | [0, 0, 0] | [0, 0, 0] | PAUSE (1.0000) | query | 0 |
| train | 79 | execute-positive-drive | [1, 0.25, 0] | [0, 0, 0] | PAUSE (1.0000) | query | 46.218 |
| train | 79 | retry-negative-drive | [-1, -0.25, 0] | [0, 0, 0] | PAUSE (1.0000) | query | 47.4392 |
| train | 79 | query-neutral-drive | [0, 0, 0] | [0, 0, 0] | PAUSE (1.0000) | query | 0 |
| train | 80 | execute-positive-drive | [1, 0.25, 0] | [0, 0, 0] | PAUSE (1.0000) | query | 24.5663 |
| train | 80 | retry-negative-drive | [-1, -0.25, 0] | [0, 0, 0] | PAUSE (1.0000) | query | 23.0728 |
| train | 80 | query-neutral-drive | [0, 0, 0] | [0, 0, 0] | PAUSE (1.0000) | query | 0 |
| train | 81 | execute-positive-drive | [1, 0.25, 0] | [0, 0, 0] | PAUSE (1.0000) | query | 29.6949 |
| train | 81 | retry-negative-drive | [-1, -0.25, 0] | [0, 0, 0] | PAUSE (1.0000) | query | 30.1628 |
| train | 81 | query-neutral-drive | [0, 0, 0] | [0, 0, 0] | PAUSE (1.0000) | query | 0 |

## Notes

- fresh one-sphere network per mode/seed/stimulus sample
- infer uses frozen cunxonNetworkStepInfer; train uses paper-canonical StepTrain
- sensitivity/diversity is diagnostic evidence, not benchmark success

## Evidence boundary

This is a sensitivity diagnostic, not a benchmark win. Input sensitivity, train-mode diversity, or action changes do not prove intelligence by themselves; this probe does not prove intelligence and only shows whether cuNxon exposes enough non-flat signal to justify a richer task adapter.
