# cuNxon Aigarth target-contract supervised low-margin objective

Status: `aigarth target-contract supervised low-margin objective completed`

## Hypothesis

The low-margin readout geometry probe showed that original execute/retry stress lanes sit on the wrong side of the observed query boundary. This diagnostic tests a supervised low-margin objective: train cases include normalized low-margin target examples, while original stress_holdout and controls stay outside the fitness callback.

## Runtime

- Device: `NVIDIA GeForce RTX 5090` / compute capability `12.0`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so`
- Seeds: `[150, 151, 152]`
- Generations/population/eval steps: `16` / `32` / `24`
- Fitness variant: `target_contract_supervised_low_margin`

## Core result

- Original stress_holdout accuracy: `0.333333`
- Original stress_holdout query-collapse rate: `1.000000`
- Original stress_holdout execute/retry accuracy: `0.000000`
- Counterfactual control accuracy: `0.000000`
- Permuted control accuracy: `0.000000`
- Aggregate actions: execute=30, query=48, retry=30

## Split summaries

| Split | Samples | Accuracy | Best constant baseline | Seeds > baseline | Query collapse | Execute/retry accuracy | Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `train` | 9 | 1.000000 | 0.333333 | 3 | 0.333333 | 1.000000 | execute=3, query=3, retry=3 |
| `augmented_train` | 18 | 1.000000 | 0.333333 | 3 | 0.333333 | 1.000000 | execute=6, query=6, retry=6 |
| `supervised_low_margin_train` | 18 | 1.000000 | 0.333333 | 3 | 0.333333 | 1.000000 | execute=6, query=6, retry=6 |
| `holdout` | 9 | 1.000000 | 0.333333 | 3 | 0.333333 | 1.000000 | execute=3, query=3, retry=3 |
| `hard_holdout` | 18 | 1.000000 | 0.333333 | 3 | 0.333333 | 1.000000 | execute=6, query=6, retry=6 |
| `stress_holdout` | 18 | 0.333333 | 0.333333 | 0 | 1.000000 | 0.000000 | query=18 |
| `counterfactual_control` | 9 | 0.000000 | 0.333333 | 0 | 0.333333 | 0.000000 | execute=3, query=3, retry=3 |
| `permuted_control` | 9 | 0.000000 | 0.333333 | 0 | 0.333333 | 0.000000 | execute=3, query=3, retry=3 |

## Notes

- fresh cuNxon network/context per seed
- fitness callback sees train, augmented_train and normalized supervised_low_margin_train only
- original stress_holdout and controls are never optimized
- objective-shaping diagnostic only; not a long-sweep or intelligence claim

## Evidence boundary

This is one supervised low-margin target-objective diagnostic. Normalized supervised_low_margin_train examples are optimized, while original stress_holdout and controls are reported separately and are not generalization evidence unless they beat constant baselines.

This supervised low-margin objective is an adapter/objective diagnostic, not intelligence evidence and not generalization evidence. Original stress/control quality must beat constant baselines across fresh seeds before any useful-computation claim changes.

## Recommended next probe

- Probe id: `low_margin_target_objective_decision`
