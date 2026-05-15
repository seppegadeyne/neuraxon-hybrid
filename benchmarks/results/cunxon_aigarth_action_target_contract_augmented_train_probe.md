# cuNxon Aigarth target-contract augmented-train audit

Status: `aigarth target-contract augmented-train audit completed`

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
- Seed offsets: 112, 113, 114, 115, 116
- Fitness variant: `target_contract_augmented_train`
- Strict expected actions: execute, query, retry
- Aggregate action distribution: execute=40, query=73, retry=37
- Unexpected action count: 0
- Unexpected action rate: 0.000000
- Unexpected normalized labels: none
- Train→hard-holdout gap mean: 0.133333

## Why this probe exists

The target-contract stress audit exposed baseline-level low-margin stress holdout behavior. This follow-up keeps the signed-first-lane contract but tests a train-only objective with additional low-margin training cases: `target_contract_augmented_train` optimizes `train` plus `augmented_train` cases, while `stress_holdout`, hard holdout, and control labels stay outside the fitness callback.

## Accuracy by split

| Split | Mean accuracy | Min | Max | Seeds > best constant baseline |
| --- | ---: | ---: | ---: | ---: |
| augmented_train | 1.000000 | 1.000000 | 1.000000 | 5/5 |
| counterfactual_control | 0.066667 | 0.000000 | 0.333333 | 0/5 |
| hard_holdout | 0.866667 | 0.666667 | 1.000000 | 5/5 |
| holdout | 1.000000 | 1.000000 | 1.000000 | 5/5 |
| overall | 0.640000 | 0.600000 | 0.700000 | 5/5 |
| permuted_control | 0.000000 | 0.000000 | 0.000000 | 0/5 |
| stress_holdout | 0.333333 | 0.333333 | 0.333333 | 0/5 |
| train | 0.933333 | 0.666667 | 1.000000 | 5/5 |

## Per-seed runs

| Seed | Train | Holdout | Hard holdout | Permuted control | Overall | Actions |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 112 | 1.000000 | 1.000000 | 0.833333 | 0.000000 | 0.633333 | execute=8, query=15, retry=7 |
| 113 | 1.000000 | 1.000000 | 0.833333 | 0.000000 | 0.633333 | execute=7, query=15, retry=8 |
| 114 | 0.666667 | 1.000000 | 1.000000 | 0.000000 | 0.633333 | execute=8, query=15, retry=7 |
| 115 | 1.000000 | 1.000000 | 0.666667 | 0.000000 | 0.600000 | execute=8, query=14, retry=8 |
| 116 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.700000 | execute=9, query=14, retry=7 |

## Permuted-control leakage/oracle check

The `permuted_control` split reuses train-like inputs with rotated expected labels. It is not optimized by the fitness callback. High accuracy here would be suspicious for label/oracle leakage; low accuracy is a sanity check, not positive intelligence evidence.
Mean permuted-control accuracy: 0.000000.

## Notes

- fresh cuNxon network/context per seed
- target-contract fitness decodes with the signed-first-lane project contract
- fitness callback uses train plus augmented_train low-margin cases only
- stress_holdout, hard holdout, and control labels are never optimized
- target-contract augmented-train stress audit, not intelligence evidence

## Evidence boundary

This is an augmented-train target-contract stress audit of a tiny Aigarth/evolutionary adapter route, not intelligence evidence. Hard-holdout accuracy, strict action-label coverage, and leakage controls must remain separate from claims about broad generalization, useful learning, or production readiness.
