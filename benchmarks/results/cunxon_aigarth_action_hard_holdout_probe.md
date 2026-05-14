# cuNxon Aigarth action hard-holdout audit

Status: `aigarth hard-holdout audit completed`

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
- Seed offsets: 87, 88, 89, 90, 91
- Strict expected actions: execute, query, retry
- Aggregate action distribution: assertive=1, execute=22, query=32, retry=20
- Unexpected action count: 1
- Unexpected action rate: 0.013333
- Unexpected normalized labels: assertive
- Train→hard-holdout gap mean: 0.500000

## Why this probe exists

The Aigarth seed sweep partially repeated a tiny baseline-beating action signal, but it also produced unstable class coverage and unexpected normalized labels. This audit keeps the train-only Aigarth fitness route but expands evaluation with harder/noisier holdouts and a permuted-control leakage/oracle check before treating the route as stronger adapter evidence.

## Accuracy by split

| Split | Mean accuracy | Min | Max | Seeds > best constant baseline |
| --- | ---: | ---: | ---: | ---: |
| hard_holdout | 0.500000 | 0.333333 | 0.833333 | 2/5 |
| holdout | 0.600000 | 0.333333 | 1.000000 | 2/5 |
| overall | 0.520000 | 0.400000 | 0.733333 | 5/5 |
| permuted_control | 0.000000 | 0.000000 | 0.000000 | 0/5 |
| train | 1.000000 | 1.000000 | 1.000000 | 5/5 |

## Per-seed runs

| Seed | Train | Holdout | Hard holdout | Permuted control | Overall | Actions |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 87 | 1.000000 | 1.000000 | 0.666667 | 0.000000 | 0.666667 | assertive=1, execute=4, query=6, retry=4 |
| 88 | 1.000000 | 0.333333 | 0.333333 | 0.000000 | 0.400000 | execute=5, query=6, retry=4 |
| 89 | 1.000000 | 0.333333 | 0.333333 | 0.000000 | 0.400000 | execute=3, query=8, retry=4 |
| 90 | 1.000000 | 1.000000 | 0.833333 | 0.000000 | 0.733333 | execute=6, query=5, retry=4 |
| 91 | 1.000000 | 0.333333 | 0.333333 | 0.000000 | 0.400000 | execute=4, query=7, retry=4 |

## Permuted-control leakage/oracle check

The `permuted_control` split reuses train-like inputs with rotated expected labels. It is not optimized by the fitness callback. High accuracy here would be suspicious for label/oracle leakage; low accuracy is a sanity check, not positive intelligence evidence.
Mean permuted-control accuracy: 0.000000.

## Notes

- fresh cuNxon network/context per seed
- fitness callback still uses train cases only; hard holdout and permuted-control labels are never optimized
- strict expected actions are execute/query/retry; assertive/explore/cautious remain out-of-contract caveats
- hard-holdout audit, not intelligence evidence

## Evidence boundary

This is a stress audit of a tiny Aigarth/evolutionary adapter route, not intelligence evidence. Hard-holdout accuracy, strict action-label coverage, and leakage controls must remain separate from claims about broad generalization, useful learning, or production readiness.
