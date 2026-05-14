# cuNxon input-port proxy target probe

Status: `input-proxy target probe viable`

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Train epochs: 12
- Train steps per case: 32
- Eval steps per case: 32
- Cases: 6
- Target-proxy input ports: [3, 4, 5]
- Motor readout ports: [11, 12, 13]

## Why this probe exists

The source-semantics audit found that output-neuron teacher forcing is not supported by the public cuNxon step-input path. This probe moves the target drive to supported input-class external drive ports, then separately reads the motor output ports without target drive during evaluation.

## Accuracy and proxy alignment

| Split | Accuracy | Mean target-proxy alignment |
| --- | ---: | ---: |
| holdout | 0.333333 | 0.000000 |
| overall | 0.333333 | 0.500000 |
| train | 0.333333 | 1.000000 |

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

| Case | Split | Expected | Target proxy | Motor readout | Decoded | Outcome | Proxy alignment | Energy |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| execute-train | train | execute | [+1, 0, 0] | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 1.000000 | 42.7318 |
| retry-train | train | retry | [-1, 0, 0] | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 1.000000 | 42.7318 |
| query-train | train | query | [0, 0, 0] | [0, 0, 0] | PAUSE (query, 1.0000) | success | 1.000000 | 42.7318 |
| execute-holdout-noisy | holdout | execute | [] | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 0.000000 | 42.7318 |
| retry-holdout-noisy | holdout | retry | [] | [0, 0, 0] | PAUSE (query, 1.0000) | failure | 0.000000 | 42.7318 |
| query-holdout-low-drive | holdout | query | [] | [0, 0, 0] | PAUSE (query, 1.0000) | success | 0.000000 | 42.7318 |

## Notes

- single sphere with sensory inputs plus input-class target proxy ports
- training uses StepTrain and expected-action neuromodulator pulses; evaluation uses StepInfer without target-proxy drive
- proxy alignment only tests observable supported input drive; motor holdout accuracy must beat baselines before any adapter claim

## Evidence boundary

This is an interface/adapter diagnostic and not a desired-output/error-channel claim. Seeing a target value on input-class proxy neurons only proves that the supported external-drive route is observable; useful learning would still require motor-output accuracy to beat trivial baselines on holdout cases. This probe does not prove intelligence, generalization, or useful learning by itself.
