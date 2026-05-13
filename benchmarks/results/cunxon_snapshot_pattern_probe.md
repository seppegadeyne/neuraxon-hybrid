# cuNxon hidden-state/snapshot + pattern store/recall probe

Status: `snapshot-pattern probe viable`

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Pattern present steps: 30
- Recall settle steps: 20
- Pattern count after store: 2
- Pattern count after clear: 0
- Recall Hamming distance: 0

## Why this probe exists

The one-sphere action and sensitivity probes mostly exposed flat `query` readouts. This diagnostic inspects richer cuNxon surfaces before building another policy: full sphere hidden-state/snapshot channels and the host-side pattern store/recall API.

## Snapshot observations

| Phase | Neurons | Active | Neutral | mean abs U | mean abs h | mean abs s_tilde | mean firing | mean astro | Energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| after-finalize | 48 | 0 | 48 | 0 | 0 | 0 | 0.2 | 0 | 0 |
| after-pattern-store | 48 | 8 | 40 | 0.740692 | 0.0337361 | 0.740572 | 0.170241 | 0.0429132 | 739.579 |
| after-recall | 48 | 2 | 46 | 0.565872 | 0.0163495 | 0.568494 | 0.128533 | 0.0635807 | 774.765 |

## Pattern recall samples

| Pattern | Mask fraction | Readout | Active | Signed sum |
| --- | ---: | --- | ---: | ---: |
| alpha | 0.5 | [0, 0, 0, 0, 0, 0, 0, 0] | 0 | 0 |
| beta | 0.5 | [0, 0, 0, 0, 0, 0, 0, 0] | 0 | 0 |

## Notes

- cunxonSphereSnapshot exposed full-neuron state channels
- pattern store/list/recall/clear APIs are callable from ctypes
- pattern recall shape/signal is diagnostic evidence, not decision-quality evidence

## Evidence boundary

Snapshot activity or pattern recall shape does not prove intelligence, generalization, or useful task learning. It only tells us whether cuNxon exposes hidden-state and pattern-memory signals that are worth testing in a richer adapter.
