# cuNxon multi-sphere/action adapter probe

Status: `multi-sphere action probe viable`

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Sphere count: 3
- Train steps per training case: 24
- Eval steps per case: 16
- Cases: 6

## Adapter and holdout results

| Split | cuNxon adapter accuracy |
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

| Case | Split | Expected | Motor readout | Decoded | Outcome | Baselines | Energy |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| execute-train | train | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | always_execute=execute, always_query=query, always_retry=retry | 204.15 |
| retry-train | train | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | always_execute=execute, always_query=query, always_retry=retry | 207.416 |
| query-train | train | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | always_execute=execute, always_query=query, always_retry=retry | 210.185 |
| execute-holdout-noisy | holdout | execute | [0, 0, 0] | PAUSE (query, 1.0000) | failure | always_execute=execute, always_query=query, always_retry=retry | 212.861 |
| retry-holdout-noisy | holdout | retry | [0, 0, 0] | PAUSE (query, 1.0000) | failure | always_execute=execute, always_query=query, always_retry=retry | 216.487 |
| query-holdout-low-drive | holdout | query | [0, 0, 0] | PAUSE (query, 1.0000) | success | always_execute=execute, always_query=query, always_retry=retry | 219.411 |

## Notes

- three-sphere sensory-to-association-to-motor topology completed
- motor readout is decoded through the existing ActionDecoder/action contract
- holdout and trivial-baseline comparison is required before any adapter claim

## Evidence boundary

This multi-sphere/action adapter is useful only if it beats trivial baselines on holdout cases. Runtime viability, inter-sphere routing, or a single positive case does not prove intelligence, generalization, or robust learning.
