# cuNxon Aigarth strict-label action audit

Status: `aigarth strict-label action audit completed`

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Generations per seed: 16
- Population size: 32
- Eval steps per case: 24
- Readout ids: 35, 36, 37
- Seed offsets: 92, 93, 94, 95, 96
- Fitness variant: `strict_label_margin`
- Strict expected actions: execute, query, retry
- Aggregate action distribution: assertive=5, execute=18, query=31, retry=21
- Unexpected action count: 5
- Unexpected action rate: 0.066667
- Unexpected normalized labels: assertive
- Train→hard-holdout gap mean: 0.300000

## Why this probe exists

The hard-holdout audit left one out-of-contract normalized label and a large train-to-hard-holdout gap. This audit changes the train-only Aigarth fitness hypothesis: `strict_label_margin` rewards expected execute/query/retry outcomes and penalizes out-of-contract normalized labels such as assertive, while keeping holdout, hard-holdout, and permuted-control labels outside the fitness callback.

## Accuracy by split

| Split | Mean accuracy | Min | Max | Seeds > best constant baseline |
| --- | ---: | ---: | ---: | ---: |
| hard_holdout | 0.700000 | 0.333333 | 1.000000 | 4/5 |
| holdout | 0.733333 | 0.333333 | 1.000000 | 3/5 |
| overall | 0.626667 | 0.400000 | 0.800000 | 5/5 |
| permuted_control | 0.000000 | 0.000000 | 0.000000 | 0/5 |
| train | 1.000000 | 1.000000 | 1.000000 | 5/5 |

## Per-seed runs

| Seed | Train | Holdout | Hard holdout | Permuted control | Overall | Actions |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 92 | 1.000000 | 1.000000 | 0.833333 | 0.000000 | 0.733333 | execute=4, query=6, retry=5 |
| 93 | 1.000000 | 0.333333 | 0.500000 | 0.000000 | 0.466667 | assertive=3, execute=2, query=7, retry=3 |
| 94 | 1.000000 | 1.000000 | 0.833333 | 0.000000 | 0.733333 | execute=4, query=6, retry=5 |
| 95 | 1.000000 | 0.333333 | 0.333333 | 0.000000 | 0.400000 | assertive=2, execute=3, query=7, retry=3 |
| 96 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.800000 | execute=5, query=5, retry=5 |

## Permuted-control leakage/oracle check

The `permuted_control` split reuses train-like inputs with rotated expected labels. It is not optimized by the fitness callback. High accuracy here would be suspicious for label/oracle leakage; low accuracy is a sanity check, not positive intelligence evidence.
Mean permuted-control accuracy: 0.000000.

## Notes

- fresh cuNxon network/context per seed
- strict-label fitness penalizes out-of-contract normalized labels
- fitness callback uses train cases only; holdout, hard-holdout and permuted-control labels are never optimized
- strict-label audit, not intelligence evidence

## Evidence boundary

This is a strict-label stress audit of a tiny Aigarth/evolutionary adapter route, not intelligence evidence. Hard-holdout accuracy, strict action-label coverage, and leakage controls must remain separate from claims about broad generalization, useful learning, or production readiness.
