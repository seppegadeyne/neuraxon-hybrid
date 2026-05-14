# cuNxon resident task-coupled action probe

Status: `resident action probe viable`

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Sphere count: 3
- Train epochs: 12
- Train steps per case: 128
- Eval steps per case: 64
- Cases: 72
- Unique motor readouts: 1

## Why this probe exists

The previous four-hour VRAM-resident run kept one small infer-only network alive but did not include task-coupled scoring. This probe keeps the same cuNxon network/context resident across repeated train/eval task epochs, decodes motor readout through the existing action contract, and compares every result with trivial constant-action baselines.

## Accuracy by resident epoch

| Epoch | Accuracy |
| ---: | ---: |
| 1 | 0.333333 |
| 2 | 0.333333 |
| 3 | 0.333333 |
| 4 | 0.333333 |
| 5 | 0.333333 |
| 6 | 0.333333 |
| 7 | 0.333333 |
| 8 | 0.333333 |
| 9 | 0.333333 |
| 10 | 0.333333 |
| 11 | 0.333333 |
| 12 | 0.333333 |

## Accuracy by split

| Split | Accuracy |
| --- | ---: |
| holdout | 0.333333 |
| overall | 0.333333 |
| train | 0.333333 |

## Trivial baselines

| Baseline | Split | Accuracy |
| --- | --- | ---: |
| always_execute | holdout | 0.333333 |
| always_execute | overall | 0.333333 |
| always_execute | train | 0.333333 |
| always_query | holdout | 0.333333 |
| always_query | overall | 0.333333 |
| always_query | train | 0.333333 |
| always_retry | holdout | 0.333333 |
| always_retry | overall | 0.333333 |
| always_retry | train | 0.333333 |

## Cases

| Epoch | Case | Split | Expected | Motor readout | Decoded | Outcome | Motor active | Energy | Elapsed ms |
| ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | execute-train | train | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 176.809 |
| 1 | retry-train | train | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 184.695 |
| 1 | query-train | train | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 192.549 |
| 1 | execute-holdout-noisy | holdout | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 200.296 |
| 1 | retry-holdout-noisy | holdout | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 3 | 209.225 | 208.065 |
| 1 | query-holdout-low-drive | holdout | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 217.562 |
| 2 | execute-train | train | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 290.486 |
| 2 | retry-train | train | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 298.373 |
| 2 | query-train | train | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 306.193 |
| 2 | execute-holdout-noisy | holdout | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 314.007 |
| 2 | retry-holdout-noisy | holdout | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 3 | 209.225 | 321.821 |
| 2 | query-holdout-low-drive | holdout | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 329.786 |
| 3 | execute-train | train | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 399.212 |
| 3 | retry-train | train | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 407.151 |
| 3 | query-train | train | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 414.963 |
| 3 | execute-holdout-noisy | holdout | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 422.823 |
| 3 | retry-holdout-noisy | holdout | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 3 | 209.225 | 430.731 |
| 3 | query-holdout-low-drive | holdout | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 438.516 |
| 4 | execute-train | train | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 514.225 |
| 4 | retry-train | train | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 522.150 |
| 4 | query-train | train | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 529.939 |
| 4 | execute-holdout-noisy | holdout | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 537.756 |
| 4 | retry-holdout-noisy | holdout | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 3 | 209.225 | 545.781 |
| 4 | query-holdout-low-drive | holdout | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 553.631 |
| 5 | execute-train | train | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 624.348 |
| 5 | retry-train | train | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 632.288 |
| 5 | query-train | train | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 640.122 |
| 5 | execute-holdout-noisy | holdout | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 647.973 |
| 5 | retry-holdout-noisy | holdout | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 3 | 209.225 | 655.961 |
| 5 | query-holdout-low-drive | holdout | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 663.806 |
| 6 | execute-train | train | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 733.695 |
| 6 | retry-train | train | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 741.360 |
| 6 | query-train | train | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 749.060 |
| 6 | execute-holdout-noisy | holdout | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 756.817 |
| 6 | retry-holdout-noisy | holdout | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 764.567 |
| 6 | query-holdout-low-drive | holdout | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 772.355 |
| 7 | execute-train | train | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 842.266 |
| 7 | retry-train | train | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 850.056 |
| 7 | query-train | train | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 857.699 |
| 7 | execute-holdout-noisy | holdout | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 865.483 |
| 7 | retry-holdout-noisy | holdout | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 873.149 |
| 7 | query-holdout-low-drive | holdout | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 880.851 |
| 8 | execute-train | train | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 951.028 |
| 8 | retry-train | train | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 958.864 |
| 8 | query-train | train | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 966.843 |
| 8 | execute-holdout-noisy | holdout | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 974.593 |
| 8 | retry-holdout-noisy | holdout | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 982.408 |
| 8 | query-holdout-low-drive | holdout | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 990.256 |
| 9 | execute-train | train | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 1060.757 |
| 9 | retry-train | train | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 1068.746 |
| 9 | query-train | train | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 1076.704 |
| 9 | execute-holdout-noisy | holdout | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 1084.548 |
| 9 | retry-holdout-noisy | holdout | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 1092.267 |
| 9 | query-holdout-low-drive | holdout | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 1100.086 |
| 10 | execute-train | train | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 1170.148 |
| 10 | retry-train | train | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 1178.054 |
| 10 | query-train | train | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 1186.013 |
| 10 | execute-holdout-noisy | holdout | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 1193.991 |
| 10 | retry-holdout-noisy | holdout | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 1201.941 |
| 10 | query-holdout-low-drive | holdout | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 1209.768 |
| 11 | execute-train | train | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 1279.752 |
| 11 | retry-train | train | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 1287.482 |
| 11 | query-train | train | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 1295.692 |
| 11 | execute-holdout-noisy | holdout | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 1305.314 |
| 11 | retry-holdout-noisy | holdout | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 1313.277 |
| 11 | query-holdout-low-drive | holdout | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 1321.057 |
| 12 | execute-train | train | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 1391.140 |
| 12 | retry-train | train | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 1398.947 |
| 12 | query-train | train | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 1406.577 |
| 12 | execute-holdout-noisy | holdout | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 1414.410 |
| 12 | retry-holdout-noisy | holdout | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 2 | 209.225 | 1422.278 |
| 12 | query-holdout-low-drive | holdout | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0 | 209.225 | 1430.300 |

## Notes

- same three-sphere cuNxon network/context remains resident across all epochs
- training uses StepTrain plus expected-action neuromodulator pulses; evaluation uses StepInfer without target labels
- holdout accuracy must beat trivial baselines before any adapter claim

## Evidence boundary

This is task-coupled runtime/adapter evidence, not an intelligence claim. A same-process resident run, action decoding, or one positive case does not prove intelligence, useful learning, or generalization. The holdout accuracy must beat trivial baselines before this route can be treated as useful adapter evidence.
