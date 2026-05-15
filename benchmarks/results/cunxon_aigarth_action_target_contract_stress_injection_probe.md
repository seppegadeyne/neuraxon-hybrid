# cuNxon Aigarth target-contract stress-injection audit

Status: `aigarth target-contract stress-injection audit completed`

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
- Seed offsets: 137, 138, 139, 140, 141
- Fitness variant: `target_contract_stress_injection`
- Strict expected actions: execute, query, retry
- Aggregate action distribution: execute=40, query=100, retry=40
- Unexpected action count: 0
- Unexpected action rate: 0.000000
- Unexpected normalized labels: none
- Train→hard-holdout gap mean: 0.133333

## Why this probe exists

The criticality/decoder separation result points to low-margin execute/retry stress query-collapse. This diagnostic changes the question from generalization to an upper-bound: `target_contract_stress_injection` deliberately includes duplicated low-margin `stress_train` cases in the Aigarth fitness callback, while reporting the original `stress_holdout` and controls separately. This leaks stress-like labels into optimization and is therefore adapter-capability/debugging evidence only, not generalization evidence.

## Accuracy by split

| Split | Mean accuracy | Min | Max | Seeds > best constant baseline |
| --- | ---: | ---: | ---: | ---: |
| augmented_train | 1.000000 | 1.000000 | 1.000000 | 5/5 |
| counterfactual_control | 0.000000 | 0.000000 | 0.000000 | 0/5 |
| hard_holdout | 0.866667 | 0.666667 | 1.000000 | 5/5 |
| holdout | 1.000000 | 1.000000 | 1.000000 | 5/5 |
| overall | 0.588889 | 0.555556 | 0.611111 | 5/5 |
| permuted_control | 0.000000 | 0.000000 | 0.000000 | 0/5 |
| stress_holdout | 0.333333 | 0.333333 | 0.333333 | 0/5 |
| stress_train | 0.333333 | 0.333333 | 0.333333 | 0/5 |
| train | 1.000000 | 1.000000 | 1.000000 | 5/5 |

## Per-seed runs

| Seed | Train | Holdout | Hard holdout | Permuted control | Overall | Actions |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 137 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.611111 | execute=8, query=20, retry=8 |
| 138 | 1.000000 | 1.000000 | 0.666667 | 0.000000 | 0.555556 | execute=8, query=20, retry=8 |
| 139 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.611111 | execute=8, query=20, retry=8 |
| 140 | 1.000000 | 1.000000 | 0.666667 | 0.000000 | 0.555556 | execute=8, query=20, retry=8 |
| 141 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.611111 | execute=8, query=20, retry=8 |

## Permuted-control leakage/oracle check

The `permuted_control` split reuses train-like inputs with rotated expected labels. It is not optimized by the fitness callback. High accuracy here would be suspicious for label/oracle leakage; low accuracy is a sanity check, not positive intelligence evidence.
Mean permuted-control accuracy: 0.000000.

## Notes

- fresh cuNxon network/context per seed
- target-contract fitness decodes with the signed-first-lane project contract
- fitness callback deliberately includes duplicated stress_train low-margin cases
- stress_holdout is the original stress split, but stress-like labels are optimized
- upper-bound/debugging diagnostic only; not generalization or intelligence evidence

## Evidence boundary

This is a stress-injection upper-bound diagnostic for a tiny Aigarth/evolutionary adapter route, not intelligence evidence. Hard-holdout accuracy, strict action-label coverage, and leakage controls must remain separate from claims about broad generalization, useful learning, or production readiness.
