# cuNxon stress stimulus geometry audit

Status: `stress stimulus geometry audit completed`

## Hypothesis

After stress-label injection failed, the next question is whether the low-margin execute/retry stress vectors are separable under the signed-first-lane target-contract route at all, or whether they are too weak/neutral and therefore collapse to `query` even when included in the Aigarth fitness callback.

## Source artifact

- Source: `benchmarks/results/cunxon_aigarth_action_target_contract_stress_injection_probe.json`
- Device: `NVIDIA GeForce RTX 5090` / compute capability `12.0`
- Seeds: `[137, 138, 139, 140, 141]`
- Fitness variant: `target_contract_stress_injection`

## Core result

- Samples analyzed: `180`
- stress_train query-collapse rate=1.000000
- stress_holdout query-collapse rate=1.000000
- Stress execute/retry accuracy: `0.000000`
- Augmented-train execute/retry accuracy: `1.000000`
- Mean abs-sum drive: stress_train `0.353333`, stress_holdout `0.353333`, augmented_train `0.433333`
- Mean L2 norm: stress_train `0.224275`, stress_holdout `0.224275`, augmented_train `0.300793`

| Split | Samples | Accuracy | Query collapse | Mean abs-sum | Mean L2 norm | Actions |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `augmented_train` | 30 | 1.000000 | 0.333333 | 0.433333 | 0.300793 | execute=10, query=10, retry=10 |
| `counterfactual_control` | 15 | 0.000000 | 0.333333 | 0.766667 | 0.577345 | execute=5, query=5, retry=5 |
| `hard_holdout` | 30 | 0.866667 | 0.333333 | 0.658333 | 0.423794 | execute=10, query=10, retry=10 |
| `holdout` | 15 | 1.000000 | 0.333333 | 0.766667 | 0.577345 | execute=5, query=5, retry=5 |
| `permuted_control` | 15 | 0.000000 | 0.333333 | 0.833333 | 0.687184 | execute=5, query=5, retry=5 |
| `stress_holdout` | 30 | 0.333333 | 1.000000 | 0.353333 | 0.224275 | query=30 |
| `stress_train` | 30 | 0.333333 | 1.000000 | 0.353333 | 0.224275 | query=30 |
| `train` | 15 | 1.000000 | 0.333333 | 0.833333 | 0.687184 | execute=5, query=5, retry=5 |

## Stress case detail

| Split | Case | Expected | L2 norm | Abs-sum | Accuracy | Query collapse | Readouts |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `stress_train` | `execute-stress-low-margin-train-injected` | `execute` | 0.230651 | 0.380000 | 0.000000 | 1.000000 | [0, 0, 0]=5 |
| `stress_train` | `execute-stress-near-neutral-train-injected` | `execute` | 0.187617 | 0.320000 | 0.000000 | 1.000000 | [0, 0, 0]=5 |
| `stress_train` | `query-stress-negative-positive-train-injected` | `query` | 0.254558 | 0.360000 | 1.000000 | 1.000000 | [0, 0, 0]=5 |
| `stress_train` | `query-stress-positive-negative-train-injected` | `query` | 0.254558 | 0.360000 | 1.000000 | 1.000000 | [0, 0, 0]=5 |
| `stress_train` | `retry-stress-low-margin-train-injected` | `retry` | 0.230651 | 0.380000 | 0.000000 | 1.000000 | [0, 0, 0]=5 |
| `stress_train` | `retry-stress-near-neutral-train-injected` | `retry` | 0.187617 | 0.320000 | 0.000000 | 1.000000 | [0, 0, 0]=5 |
| `stress_holdout` | `execute-stress-low-margin` | `execute` | 0.230651 | 0.380000 | 0.000000 | 1.000000 | [0, 0, 0]=5 |
| `stress_holdout` | `execute-stress-near-neutral` | `execute` | 0.187617 | 0.320000 | 0.000000 | 1.000000 | [0, 0, 0]=5 |
| `stress_holdout` | `query-stress-negative-positive` | `query` | 0.254558 | 0.360000 | 1.000000 | 1.000000 | [0, 0, 0]=5 |
| `stress_holdout` | `query-stress-positive-negative` | `query` | 0.254558 | 0.360000 | 1.000000 | 1.000000 | [0, 0, 0]=5 |
| `stress_holdout` | `retry-stress-low-margin` | `retry` | 0.230651 | 0.380000 | 0.000000 | 1.000000 | [0, 0, 0]=5 |
| `stress_holdout` | `retry-stress-near-neutral` | `retry` | 0.187617 | 0.320000 | 0.000000 | 1.000000 | [0, 0, 0]=5 |

## Interpretation

The current stress bottleneck is consistent with low-margin stimulus/readout separability failure: injected stress_train and original stress_holdout both collapse to query on every sample, while augmented_train execute/retry cases remain separable.

The successful `augmented_train` execute/retry cases have stronger signed drive (`abs_sum` mean `0.433333`, execute/retry L2 around `0.379747`) than the injected `stress_train` and original `stress_holdout` cases (`abs_sum` mean `0.353333`, execute/retry L2 around `0.209134`). In this artifact, every stress execute/retry case produced readout `[0, 0, 0]` and decoded to `query`, while every query stress case also decoded to `query` and therefore scored correctly. That makes the stress split look baseline-level for a very specific reason: neutral readout is rewarded for query cases and destructive for low-margin execute/retry cases.

## Recommended next probe

- GitHub issue: https://github.com/sisutuulenisa/neuraxon-hybrid/issues/87
- Probe id: `stress_amplitude_ladder_probe`
- Run a bounded live amplitude ladder over the same low-margin stress vectors before adding more task complexity or longer runtime.
- Keep label-injected stress evidence separated from original `stress_holdout` and controls.

## Evidence boundary

This is a post-hoc geometry/separability audit over a live RTX 5090 artifact. It is useful diagnostic evidence, not intelligence evidence, not a generalization claim, and not proof that scaling will solve the bottleneck until a bounded live probe beats constant baselines on stress/control splits.
