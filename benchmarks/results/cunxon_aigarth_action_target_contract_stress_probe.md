# cuNxon Aigarth target-contract stress audit

Status: `aigarth target-contract stress audit completed`

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
- Seed offsets: 107, 108, 109, 110, 111
- Fitness variant: `target_contract_margin`
- Strict expected actions: execute, query, retry
- Aggregate action distribution: execute=28, query=68, retry=24
- Unexpected action count: 0
- Unexpected action rate: 0.000000
- Unexpected normalized labels: none
- Train→hard-holdout gap mean: 0.466667

## Why this probe exists

The target-contract audit is the strongest cuNxon action lane so far, but the task is tiny and one seed stayed baseline-level on hard-holdout. This audit keeps the same train-only `target_contract_margin` objective and adds harder/noisier and counterfactual splits (`stress_holdout` and `counterfactual_control`) before treating the route as stronger adapter evidence.

## Accuracy by split

| Split | Mean accuracy | Min | Max | Seeds > best constant baseline |
| --- | ---: | ---: | ---: | ---: |
| counterfactual_control | 0.266667 | 0.000000 | 0.333333 | 0/5 |
| hard_holdout | 0.533333 | 0.333333 | 1.000000 | 3/5 |
| holdout | 0.666667 | 0.333333 | 1.000000 | 3/5 |
| overall | 0.458333 | 0.375000 | 0.583333 | 5/5 |
| permuted_control | 0.000000 | 0.000000 | 0.000000 | 0/5 |
| stress_holdout | 0.333333 | 0.333333 | 0.333333 | 0/5 |
| train | 1.000000 | 1.000000 | 1.000000 | 5/5 |

## Per-seed runs

| Seed | Train | Holdout | Hard holdout | Permuted control | Overall | Actions |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 107 | 1.000000 | 0.333333 | 0.333333 | 0.000000 | 0.375000 | execute=6, query=12, retry=6 |
| 108 | 1.000000 | 1.000000 | 0.500000 | 0.000000 | 0.500000 | execute=5, query=14, retry=5 |
| 109 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.583333 | execute=6, query=12, retry=6 |
| 110 | 1.000000 | 0.333333 | 0.333333 | 0.000000 | 0.375000 | execute=6, query=13, retry=5 |
| 111 | 1.000000 | 0.666667 | 0.500000 | 0.000000 | 0.458333 | execute=5, query=17, retry=2 |

## Permuted-control leakage/oracle check

The `permuted_control` split reuses train-like inputs with rotated expected labels. It is not optimized by the fitness callback. High accuracy here would be suspicious for label/oracle leakage; low accuracy is a sanity check, not positive intelligence evidence.
Mean permuted-control accuracy: 0.000000.

## Notes

- fresh cuNxon network/context per seed
- target-contract fitness decodes with the signed-first-lane project contract
- adds stress_holdout low-margin cases and counterfactual_control rotated-label cases
- fitness callback uses train cases only; all holdout and control labels are never optimized
- target-contract stress audit, not intelligence evidence

## Evidence boundary

This is a harder target-contract stress audit of a tiny Aigarth/evolutionary adapter route, not intelligence evidence. Hard-holdout accuracy, strict action-label coverage, and leakage controls must remain separate from claims about broad generalization, useful learning, or production readiness.
