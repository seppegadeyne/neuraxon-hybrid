# cuNxon long-horizon learning probe

Status: `long-horizon probe viable`

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Total steps: 65536
- Sample interval: 8192
- Samples: 8
- Unique readouts: 1
- readout changes: 0
- Energy delta: 1.11955e+06

## Long-horizon learning caveat

Neuraxon should be treated as a brain-like, time-dependent learning system: short smoke tests are insufficient for judging whether it learns. This probe keeps one cuNxon network instance active for a longer step horizon and samples readout/energy over time. It is still only runtime/dynamics evidence, not a decision-quality benchmark.

## Samples

| Step | Readout | Energy | Elapsed ms |
| ---: | --- | ---: | ---: |
| 8192 | [0, 0, 0] | 157619 | 513.429 |
| 16384 | [0, 0, 0] | 316491 | 914.622 |
| 24576 | [0, 0, 0] | 475921 | 1318.067 |
| 32768 | [0, 0, 0] | 635720 | 1714.106 |
| 40960 | [0, 0, 0] | 795779 | 2118.218 |
| 49152 | [0, 0, 0] | 956073 | 2529.347 |
| 57344 | [0, 0, 0] | 1.11654e+06 | 2927.323 |
| 65536 | [0, 0, 0] | 1.27717e+06 | 3325.790 |

## Notes

- continuous one-sphere long-horizon probe completed
- short smoke tests are insufficient for learning claims
- inter-sphere Python demo remains separate from this probe

## Evidence boundary

This long-horizon probe does not prove intelligence, generalization, or useful learning by itself. It only checks whether a continuous cuNxon run produces valid trinary readouts and observable dynamics over a longer active window.
