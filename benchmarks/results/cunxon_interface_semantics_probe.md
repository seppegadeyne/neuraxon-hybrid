# cuNxon interface semantics probe

Status: `interface semantics probe viable`

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Steps: 3
- Samples: 4

## Why this probe exists

previous multi-sphere probes stayed flat, so this slice checks whether cuNxon interfaces use absolute neuron indices rather than relative output-block indices before building another supervised motor-target adapter.

## Same-sphere readout-port samples

| Mapping | Port IDs | Neuron class | Readout | Snapshot slice | Matches snapshot | Active | Signed sum |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| relative-input-readout | [0, 1, 2, 3] | input | [+1, -1, +1, +1] | [+1, -1, +1, +1] | True | 4 | 2 |
| absolute-output-readout | [8, 9, 10, 11] | output | [-1, 0, +1, 0] | [-1, 0, +1, 0] | True | 2 | 0 |

## Relay samples

| Mapping | Source port IDs | Source class | Source readout | Downstream input readout | Downstream active | Downstream energy |
| --- | --- | --- | --- | --- | ---: | ---: |
| input-neuron-relay | [0, 1, 2, 3] | input | [+1, -1, +1, +1] | [+1, -1, -1, -1] | 4 | 22.4048 |
| output-neuron-relay | [8, 9, 10, 11] | output | [-1, 0, +1, 0] | [0, 0, 0, 0] | 0 | 18.8491 |

## Notes

- same-sphere readouts were compared against full-sphere snapshot slices
- relay samples use identical source inputs but alternate input-vs-output source ports
- absolute output ports are input_count + hidden_count + output offset for this 4/4/4 probe

## Evidence boundary

This probe only checks cuNxon C API interface/readout-port semantics. It does not prove intelligence, task learning, action quality, or generalization. A supervised motor-target adapter would still need holdout cases and trivial baselines before any usefulness claim.
