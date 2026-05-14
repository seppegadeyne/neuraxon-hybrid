# cuNxon Aigarth action seed sweep

Status: `aigarth action seed sweep completed`

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
- Seed offsets: 82, 83, 84, 85, 86
- Aggregate action distribution: assertive=4, execute=3, query=16, retry=7
- Coverage note: all three action labels appeared at least once; unexpected normalized labels: assertive

## Why this probe exists

The single-seed Aigarth action probe was the first cuNxon action lane that beat constant-action baselines, but it still failed the retry class and used one engineered seed. This sweep reruns the same train-only Aigarth fitness protocol with a fresh cuNxon network/context per seed to test repeatability before treating the result as stronger adapter evidence.

## Seed repeatability

| Split | Mean accuracy | Min | Max | Seeds > best constant baseline |
| --- | ---: | ---: | ---: | ---: |
| holdout | 0.533333 | 0.333333 | 0.666667 | 3/5 |
| overall | 0.600000 | 0.500000 | 0.833333 | 5/5 |
| train | 0.666667 | 0.333333 | 1.000000 | 4/5 |

## Per-seed runs

| Seed | Train | Holdout | Overall | Unique readouts | Actions | Final train score |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 82 | 0.333333 | 0.666667 | 0.500000 | 4 | assertive=2, query=3, retry=1 | 1.250000 |
| 83 | 0.666667 | 0.666667 | 0.666667 | 5 | assertive=1, execute=1, query=3, retry=1 | 0.888889 |
| 84 | 0.666667 | 0.333333 | 0.500000 | 5 | query=4, retry=2 | 1.250000 |
| 85 | 1.000000 | 0.666667 | 0.833333 | 5 | execute=1, query=3, retry=2 | 1.111111 |
| 86 | 0.666667 | 0.333333 | 0.500000 | 5 | assertive=1, execute=1, query=3, retry=1 | 0.861111 |

## Notes

- fresh cuNxon network/context per seed
- fitness callback still uses train cases only; holdout labels are never optimized
- repeatability audit, not intelligence evidence

## Evidence boundary

This is a repeatability audit, not intelligence evidence. A seed sweep can show whether a small Aigarth/evolutionary adapter result is stable or brittle, but it does not by itself prove broad generalization, useful learning, or production readiness. Partial action coverage remains a caveat until all expected action labels are learned and verified on harder holdouts.
