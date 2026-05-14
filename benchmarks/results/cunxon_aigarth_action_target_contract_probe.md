# cuNxon Aigarth target-contract action audit

Status: `aigarth target-contract action audit completed`

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
- Seed offsets: 102, 103, 104, 105, 106
- Fitness variant: `target_contract_margin`
- Strict expected actions: execute, query, retry
- Aggregate action distribution: execute=20, query=35, retry=20
- Unexpected action count: 0
- Unexpected action rate: 0.000000
- Unexpected normalized labels: none
- Train→hard-holdout gap mean: 0.333333

## Why this probe exists

The remap audit showed that a signed-first-lane decoder removes out-of-contract labels but does not improve accuracy when applied only post hoc. This audit moves that decoder contract into the train-only Aigarth fitness itself: `target_contract_margin` decodes with the signed-first-lane project contract and rewards target-readout margin while keeping holdout, hard-holdout, and permuted-control labels outside the fitness callback.

## Accuracy by split

| Split | Mean accuracy | Min | Max | Seeds > best constant baseline |
| --- | ---: | ---: | ---: | ---: |
| hard_holdout | 0.666667 | 0.333333 | 1.000000 | 4/5 |
| holdout | 0.866667 | 0.333333 | 1.000000 | 4/5 |
| overall | 0.640000 | 0.400000 | 0.800000 | 5/5 |
| permuted_control | 0.000000 | 0.000000 | 0.000000 | 0/5 |
| train | 1.000000 | 1.000000 | 1.000000 | 5/5 |

## Per-seed runs

| Seed | Train | Holdout | Hard holdout | Permuted control | Overall | Actions |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 102 | 1.000000 | 0.333333 | 0.333333 | 0.000000 | 0.400000 | execute=2, query=11, retry=2 |
| 103 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.800000 | execute=5, query=5, retry=5 |
| 104 | 1.000000 | 1.000000 | 0.666667 | 0.000000 | 0.666667 | execute=4, query=7, retry=4 |
| 105 | 1.000000 | 1.000000 | 0.666667 | 0.000000 | 0.666667 | execute=5, query=5, retry=5 |
| 106 | 1.000000 | 1.000000 | 0.666667 | 0.000000 | 0.666667 | execute=4, query=7, retry=4 |

## Permuted-control leakage/oracle check

The `permuted_control` split reuses train-like inputs with rotated expected labels. It is not optimized by the fitness callback. High accuracy here would be suspicious for label/oracle leakage; low accuracy is a sanity check, not positive intelligence evidence.
Mean permuted-control accuracy: 0.000000.

## Notes

- fresh cuNxon network/context per seed
- target-contract fitness decodes with the signed-first-lane project contract
- fitness callback uses train cases only; holdout, hard-holdout and permuted-control labels are never optimized
- target-contract audit, not intelligence evidence

## Evidence boundary

This is a target-contract stress audit of a tiny Aigarth/evolutionary adapter route, not intelligence evidence. Hard-holdout accuracy, strict action-label coverage, and leakage controls must remain separate from claims about broad generalization, useful learning, or production readiness.
