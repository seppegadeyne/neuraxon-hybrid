# cuNxon Aigarth readout semantics probe

Status: `aigarth readout semantics probe viable`

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Generations: 12
- Population size: 24
- Eval steps per class: 25

## Why this probe exists

Upstream `example_aigarth.cu` is the public supervised/evolutionary cuNxon example, but it configures readout ids `0..7` for a 4-input/32-hidden/8-output sphere. Those ids are relative to the sphere start and therefore alias input/hidden neurons, not the absolute output block `36..43`. This probe contrasts the demo-relative mapping with the absolute output mapping before treating Aigarth as a sanctioned motor/readout route for Neuraxon-Hybrid.

## Mapping results

| Mapping | Ports | Neuron class | Baseline margin | Final margin | Improvement | Pos mean | Neg mean | Pos readout | Neg readout |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| relative-demo-readout | [0, 1, 2, 3, 4, 5, 6, 7] | input/hidden alias, not output block | 1.000000 | 1.500000 | 0.500000 | 0.750000 | -0.750000 | [+1, +1, +1, +1, +1, +1, +1, -1] | [-1, -1, -1, -1, -1, -1, -1, +1] |
| absolute-output-readout | [36, 37, 38, 39, 40, 41, 42, 43] | absolute output block | 0.625000 | 1.500000 | 0.875000 | 0.750000 | -0.750000 | [+1, +1, -1, +1, +1, +1, +1, +1] | [-1, -1, +1, -1, -1, -1, -1, -1] |

## Evolution trajectories

- `relative-demo-readout` generation margins: 1.250000, 1.250000, 1.375000, 1.375000, 1.375000, 1.500000, 1.375000, 1.500000, 1.500000, 1.500000, 1.500000, 1.500000
- `absolute-output-readout` generation margins: 1.375000, 1.500000, 1.500000, 1.500000, 1.500000, 1.500000, 1.375000, 1.500000, 1.375000, 1.500000, 1.500000, 1.500000

## Notes

- contrasts upstream example_aigarth.cu readout ids 0..7 with absolute output ids 36..43
- Aigarth is evaluated as interface semantics, not as an intelligence claim
- absolute output readout must improve on its own before using this route as a motor adapter

## Evidence boundary

Aigarth margin movement is useful interface evidence, but it does not prove intelligence, task learning, or a useful motor adapter unless absolute output readouts improve against baselines and survive holdout/generalization tests. Relative demo-readout improvement alone is confounded by input/hidden aliasing.
