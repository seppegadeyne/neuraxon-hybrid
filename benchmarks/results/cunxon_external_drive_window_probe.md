# cuNxon external-drive window probe

Status: `external-drive window probe viable`

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Steps per sample: 5
- Input vector: [1, -1, 0.5, -0.5]
- Samples: 6

## Why this probe exists

The source-semantics audit and input-proxy target probe suggest that the public step-input path is an input-class direct drive, not a desired-output/error-channel. This controlled ctypes probe drives identical values through input, hidden, and output sensory-id windows in both `StepInfer` and `StepTrain`, then compares the configured readout to the full-sphere snapshot slice.

## Port-window samples

| Mode | Driven class | Sensory IDs | Readout IDs | Readout | Snapshot slice | Matches snapshot | Active | Signed sum | Energy |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| infer | input | [0, 1, 2, 3] | [0, 1, 2, 3] | [+1, -1, +1, -1] | [+1, -1, +1, -1] | True | 4 | 0 | 63.933 |
| infer | hidden | [4, 5, 6, 7] | [4, 5, 6, 7] | [0, 0, 0, 0] | [0, 0, 0, 0] | True | 0 | 0 | 0 |
| infer | output | [8, 9, 10, 11] | [8, 9, 10, 11] | [0, 0, 0, 0] | [0, 0, 0, 0] | True | 0 | 0 | 0 |
| train | input | [0, 1, 2, 3] | [0, 1, 2, 3] | [+1, -1, +1, -1] | [+1, -1, +1, -1] | True | 4 | 0 | 16.8078 |
| train | hidden | [4, 5, 6, 7] | [4, 5, 6, 7] | [0, 0, 0, 0] | [0, 0, 0, 0] | True | 0 | 0 | 0 |
| train | output | [8, 9, 10, 11] | [8, 9, 10, 11] | [0, 0, 0, 0] | [0, 0, 0, 0] | True | 0 | 0 | 0 |

## Interpretation boundary

input-class direct drive is useful for observing external stimulation, but hidden/output sensory-id windows are not a supported desired-output/error-channel unless they produce target-free motor accuracy above trivial baselines in a later task-coupled probe.

## Notes

- single 4/4/4 sphere per sample with identical external vector and changed sensory ids
- readouts are compared against full-sphere snapshot slices to validate port mapping
- runtime/interface evidence only; not a desired-output or decision-quality claim

## Evidence boundary

This probe only checks port-window drive and readout semantics. It does not prove intelligence, task learning, useful recall, action quality, or generalization.
