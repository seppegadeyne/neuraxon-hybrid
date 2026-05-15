# cuNxon Aigarth target-contract stress objective

Status: `aigarth target-contract stress objective completed`

## Hypothesis

After the stress amplitude-ladder showed that high-amplitude stress vectors are separable while original low-margin stress_holdout stays collapsed, this probe tests one target-aligned, margin-weighted objective. The goal is to preserve the scaled separability signal while checking whether original stress/control quality improves without upgrading label-injected training cases to generalization evidence.

## Runtime

- Device: `NVIDIA GeForce RTX 5090` / compute capability `12.0`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`
- Seeds: `[147, 148, 149]`
- Generations/population/eval steps: `16` / `32` / `24`
- Fitness variant: `target_contract_stress_margin_weighted`
- Stress amplitude factor: `3.0x`

## Core result

- Original stress_holdout accuracy: `0.333333`
- Original stress_holdout query-collapse rate: `1.000000`
- Original stress_holdout execute/retry accuracy: `0.000000`
- Scaled stress_holdout accuracy: `0.888889`
- Scaled stress_holdout query-collapse rate: `0.222222`
- Scaled stress_holdout execute/retry accuracy: `1.000000`
- Counterfactual control accuracy: `0.111111`
- Permuted control accuracy: `0.000000`
- Aggregate actions: execute=30, query=61, retry=35

## Split summaries

| Split | Amplitude | Samples | Accuracy | Best constant baseline | Seeds > baseline | Query collapse | Execute/retry accuracy | Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `stress_holdout` | 1.0x | 18 | 0.333333 | 0.333333 | 0 | 1.000000 | 0.000000 | query=18 |
| `stress_train_scaled_3_0x` | 3.0x | 18 | 1.000000 | 0.333333 | 3 | 0.333333 | 1.000000 | execute=6, query=6, retry=6 |
| `stress_holdout_scaled_3_0x` | 3.0x | 18 | 0.888889 | 0.333333 | 3 | 0.222222 | 1.000000 | execute=6, query=4, retry=8 |

## Notes

- fresh cuNxon network/context per seed
- fitness callback weights scaled stress_train margin more strongly than the ladder
- original stress_holdout and controls are never optimized
- objective-shaping diagnostic only; not a long-sweep or intelligence claim

## Evidence boundary

This is one label-injected target-aligned objective-shaping diagnostic. Scaled stress_train cases are optimized with extra margin weight, while original stress_holdout, controls, and scaled holdouts are reported separately. It is not intelligence evidence and not generalization evidence unless original stress/control splits beat constant baselines.

This target-aligned stress objective is an objective-shaping diagnostic, not intelligence evidence. If original stress/control quality stays at constant baselines, the next question is decoder/readout geometry rather than a longer run of the same objective.

## Recommended next probe

- Probe id: `stress_objective_decoder_geometry_followup`
