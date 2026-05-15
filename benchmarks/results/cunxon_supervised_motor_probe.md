# cuNxon supervised motor-target probe

Status: `supervised motor-target probe viable`

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Train epochs: 8
- Train steps per case: 16
- Eval steps per case: 16
- Cases: 6
- Absolute output-neuron target ports: [8, 9, 10]

## Why this probe exists

The interface-semantics probe supports absolute neuron indices for output ports. This follow-up tests whether teacher-forcing those absolute output-neuron target ports can create a motor readout that beats trivial constant-action baselines on holdout cases.

## Accuracy and Target alignment

| Split | Accuracy | Target alignment |
| --- | ---: | ---: |
| holdout | 0.333333 | 0.777778 |
| overall | 0.333333 | 0.777778 |
| train | 0.333333 | 0.777778 |

## Trivial baselines

| Baseline | Split | Accuracy |
| --- | --- | ---: |
| always_execute | holdout | 0.333333 |
| always_execute | overall | 0.333333 |
| always_execute | train | 0.333333 |
| always_query | holdout | 0.333333 |
| always_query | overall | 0.333333 |
| always_query | train | 0.333333 |
| always_retry | holdout | 0.333333 |
| always_retry | overall | 0.333333 |
| always_retry | train | 0.333333 |

## Cases

| Case | Split | Expected | Target | Teacher readout | Eval readout | Decoded | Outcome | Target alignment | Energy |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| execute-train | train | execute | [+1, 0, 0] | [0, 0, 0] | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 0.666667 | 30.7825 |
| retry-train | train | retry | [-1, 0, 0] | [0, 0, 0] | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 0.666667 | 30.7825 |
| query-train | train | query | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] | PAUSE (query, 1.0000) | success | 1.000000 | 30.7825 |
| execute-holdout-noisy | holdout | execute | [+1, 0, 0] | [] | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 0.666667 | 30.7825 |
| retry-holdout-noisy | holdout | retry | [-1, 0, 0] | [] | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 0.666667 | 30.7825 |
| query-holdout-low-drive | holdout | query | [0, 0, 0] | [] | [0, 0, 0] | PAUSE (query, 1.0000) | success | 1.000000 | 30.7825 |

## Notes

- single motor sphere with sensory inputs plus teacher-forced absolute output-neuron target ports
- training uses StepTrain and expected-action neuromodulator pulses; evaluation uses StepInfer without target drive
- holdout accuracy must beat trivial baselines before this becomes useful adapter evidence

## Evidence boundary

This supervised motor-target adapter is still diagnostic. Teacher-forcing absolute output-neuron ports, a positive target-alignment score, or a single train-case success does not prove intelligence, generalization, or useful learning unless holdout accuracy beats trivial baselines.
