# cuNxon Aigarth holdout action probe

Status: `aigarth action probe viable`

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Generations: 18
- Population size: 36
- Eval steps per case: 24
- Readout ids: 35, 36, 37
- Cases: 6
- Unique readouts: 4
- Action distribution: execute=2, query=4

## Why this probe exists

The earlier Aigarth readout-semantics probe showed that the public `cunxonNetworkAigarthStep` evolutionary surface can improve an absolute-output two-pattern margin. This probe makes that route task-coupled: the fitness callback uses train cases only, then the evolved network is evaluated on both train and holdout cases through the existing `ActionDecoder` action contract.

## Fitness trajectory

- Train-only generation scores: 0.861111, 1.166667, 1.166667, 1.250000, 1.250000, 1.250000, 1.194444, 0.527778, 1.222222, 0.861111, 1.222222, 1.138889, 1.166667, 1.166667, 1.166667, 1.138889, 1.166667, 1.194444

## Accuracy by split

| Split | Accuracy | Target alignment |
| --- | ---: | ---: |
| holdout | 0.666667 | 0.555556 |
| overall | 0.666667 | 0.555556 |
| train | 0.666667 | 0.555556 |

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

| Case | Split | Expected | Target | Readout | Decoded | Outcome | Alignment | Energy |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| execute-train | train | execute | [+1, 0, 0] | [+1, +1, -1] | PROCEED (execute, 0.6667) | success | 0.333333 | 4853.82 |
| retry-train | train | retry | [-1, 0, 0] | [-1, -1, -1] | PAUSE (query, 1.0000) | failure | 0.333333 | 5315.21 |
| query-train | train | query | [0, 0, 0] | [0, 0, 0] | PAUSE (query, 1.0000) | success | 1.000000 | 0 |
| execute-holdout-noisy | holdout | execute | [+1, 0, 0] | [0, +1, 0] | PROCEED (execute, 0.3333) | success | 0.333333 | 4348.46 |
| retry-holdout-noisy | holdout | retry | [-1, 0, 0] | [-1, -1, -1] | PAUSE (query, 1.0000) | failure | 0.333333 | 4129.49 |
| query-holdout-low-drive | holdout | query | [0, 0, 0] | [0, 0, 0] | PAUSE (query, 1.0000) | success | 1.000000 | 0 |

## Notes

- Aigarth fitness callback uses train cases only; holdout labels are not optimized
- final train and holdout readouts are decoded through the existing ActionDecoder
- holdout accuracy must beat trivial baselines before any adapter claim

## Evidence boundary

This is Aigarth/evolutionary action-readout evidence, not an intelligence claim. A train fitness improvement, callable GPU evolution loop, or isolated train success does not prove intelligence, useful learning, or generalization. The holdout accuracy must beat trivial baselines before this route can be treated as useful adapter evidence.
