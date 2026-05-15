# cuNxon Aigarth target-contract stress amplitude-ladder

Status: `aigarth target-contract stress amplitude-ladder completed`

## Hypothesis

The stress geometry audit showed that low-margin execute/retry stress vectors collapsed to `query`. This bounded stress amplitude-ladder scales those vectors to test whether stronger stimulus drive reduces query collapse under the same signed-first-lane target-contract route.

## Runtime

- Device: `NVIDIA GeForce RTX 5090` / compute capability `12.0`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`
- Seeds: `[142, 143, 144]`
- Generations/population/eval steps: `16` / `32` / `24`
- Amplitude ladder: 1.0x, 1.5x, 2.0x, 3.0x

## Core result

- Original stress_holdout accuracy: `0.333333`
- Original stress_holdout query-collapse rate: `1.000000`
- Best scaled stress_holdout accuracy: `0.833333` at `3.0x`
- Aggregate actions: execute=48, query=145, retry=41

## Amplitude split summaries

| Split | Amplitude | Samples | Accuracy | Best constant baseline | Seeds > baseline | Query collapse | Execute/retry accuracy | Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `stress_train_scaled_1_0x` | 1.0x | 18 | 0.333333 | 0.333333 | 0 | 1.000000 | 0.000000 | query=18 |
| `stress_holdout_scaled_1_0x` | 1.0x | 18 | 0.333333 | 0.333333 | 0 | 1.000000 | 0.000000 | query=18 |
| `stress_train_scaled_1_5x` | 1.5x | 18 | 0.611111 | 0.333333 | 3 | 0.611111 | 0.500000 | execute=4, query=11, retry=3 |
| `stress_holdout_scaled_1_5x` | 1.5x | 18 | 0.666667 | 0.333333 | 3 | 0.666667 | 0.500000 | execute=3, query=12, retry=3 |
| `stress_train_scaled_2_0x` | 2.0x | 18 | 0.555556 | 0.333333 | 3 | 0.777778 | 0.333333 | execute=3, query=14, retry=1 |
| `stress_holdout_scaled_2_0x` | 2.0x | 18 | 0.666667 | 0.333333 | 3 | 0.555556 | 0.583333 | execute=3, query=10, retry=5 |
| `stress_train_scaled_3_0x` | 3.0x | 18 | 0.777778 | 0.333333 | 3 | 0.333333 | 0.833333 | execute=7, query=6, retry=5 |
| `stress_holdout_scaled_3_0x` | 3.0x | 18 | 0.833333 | 0.333333 | 3 | 0.388889 | 0.833333 | execute=7, query=7, retry=4 |

## Notes

- fresh cuNxon network/context per seed
- target-contract fitness includes train, augmented_train, and scaled stress_train cases
- original stress_holdout and scaled stress_holdout splits are reported separately
- bounded amplitude ladder; not a long-sweep or intelligence claim

## Evidence boundary

This is a label-injected separability upper-bound over scaled low-margin stress vectors. Scaled stress_train cases are inside the fitness callback, so any improvement is diagnostic adapter-capability evidence, not intelligence evidence and not generalization evidence unless stress/control holdouts beat constant baselines.

## Recommended next probe

- Probe id: `target_aligned_stress_objective_followup`
