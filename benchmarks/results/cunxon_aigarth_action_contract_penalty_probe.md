# cuNxon Aigarth contract-penalty action audit

Status: `aigarth contract-penalty action audit completed`

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
- Seed offsets: 97, 98, 99, 100, 101
- Fitness variant: `strict_label_heavy_penalty`
- Strict expected actions: execute, query, retry
- Aggregate action distribution: assertive=3, execute=17, query=39, retry=16
- Unexpected action count: 3
- Unexpected action rate: 0.040000
- Unexpected normalized labels: assertive
- Train→hard-holdout gap mean: 0.533333

## Why this probe exists

The strict-label audit improved hard-holdout accuracy but still emitted out-of-contract normalized labels. This audit keeps the same train-only Aigarth fitness route and holdout/control split, but applies a heavier unexpected-label penalty before increasing task difficulty.

## Accuracy by split

| Split | Mean accuracy | Min | Max | Seeds > best constant baseline |
| --- | ---: | ---: | ---: | ---: |
| hard_holdout | 0.466667 | 0.333333 | 0.833333 | 2/5 |
| holdout | 0.466667 | 0.333333 | 1.000000 | 1/5 |
| overall | 0.480000 | 0.400000 | 0.733333 | 5/5 |
| permuted_control | 0.000000 | 0.000000 | 0.000000 | 0/5 |
| train | 1.000000 | 1.000000 | 1.000000 | 5/5 |

## Per-seed runs

| Seed | Train | Holdout | Hard holdout | Permuted control | Overall | Actions |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 97 | 1.000000 | 0.333333 | 0.333333 | 0.000000 | 0.400000 | assertive=1, execute=2, query=10, retry=2 |
| 98 | 1.000000 | 0.333333 | 0.500000 | 0.000000 | 0.466667 | assertive=1, execute=3, query=9, retry=2 |
| 99 | 1.000000 | 0.333333 | 0.333333 | 0.000000 | 0.400000 | assertive=1, execute=3, query=8, retry=3 |
| 100 | 1.000000 | 1.000000 | 0.833333 | 0.000000 | 0.733333 | execute=5, query=6, retry=4 |
| 101 | 1.000000 | 0.333333 | 0.333333 | 0.000000 | 0.400000 | execute=4, query=6, retry=5 |

## Permuted-control leakage/oracle check

The `permuted_control` split reuses train-like inputs with rotated expected labels. It is not optimized by the fitness callback. High accuracy here would be suspicious for label/oracle leakage; low accuracy is a sanity check, not positive intelligence evidence.
Mean permuted-control accuracy: 0.000000.

## Notes

- fresh cuNxon network/context per seed
- heavy contract penalty subtracts three times the unexpected-label rate
- fitness callback uses train cases only; holdout, hard-holdout and permuted-control labels are never optimized
- contract-penalty audit, not intelligence evidence

## Evidence boundary

This is a heavier contract-penalty stress audit of a tiny Aigarth/evolutionary adapter route, not intelligence evidence. Hard-holdout accuracy, strict action-label coverage, and leakage controls must remain separate from claims about broad generalization, useful learning, or production readiness.
