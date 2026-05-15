# cuNxon task-coupled action probe

Status: `task-coupled action probe viable`
Accuracy: 0.333333 (1/3)

## Source

- Upstream repo commit: `bd2242fabad08cb73dab2c4170d11fa941030e8c`
- cuNxon commit: `b4f6db85f7aff04ddb4e1078d523d514a278521b`
- Library: `/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153/libcunxon.so.0.1.0`

## GPU/runtime

- Device: NVIDIA GeForce RTX 5090
- Compute capability: 12.0
- Trial steps: 32
- Trials: 3

## Decision-quality boundary

This probe is task-coupled: each input drive has an expected benchmark action and the cuNxon readout is decoded through the existing Neuraxon-Hybrid ActionDecoder/action-contract mapping. That makes it decision-quality evidence, but flat or baseline-level results still do not prove intelligence, generalization, or useful learning. A task-coupled GPU probe does not prove intelligence unless it beats simple baselines and survives holdout tests.

## Trials

| Trial | Input | Readout | Decoded | Normalized | Expected | Outcome | Energy |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| execute-positive-drive | [1, 0.25, 0] | [0, 0, -1] | RETRY (0.3333) | retry | execute | failure | 708.582 |
| retry-negative-drive | [-1, -0.25, 0] | [0, 0, 0] | PAUSE (1.0000) | query | retry | failure | 1144.26 |
| query-neutral-drive | [0, 0, 0] | [0, 0, 0] | PAUSE (1.0000) | query | query | success | 1356.27 |

## Notes

- sequential one-sphere task-coupled probe completed
- readouts are decoded with the existing ActionDecoder/action-contract mapping
- flat or baseline-level accuracy is negative evidence, not intelligence evidence

## Evidence boundary

A GPU-backed action probe only becomes interesting if it beats simple baselines and survives richer temporal/generalization tests. A flat or baseline-level readout should be treated as a negative/diagnostic result, not as evidence for Neuraxon intelligence.
