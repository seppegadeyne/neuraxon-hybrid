# cuNxon Aigarth action remap audit

Status: `aigarth action remap audit completed`

## Scope

This is a post-hoc diagnostic over existing live cuNxon Aigarth action artifacts. It does not start a new GPU evolution run and does not create new cuNxon learning evidence. Its purpose is to isolate whether the out-of-contract `assertive` labels came from the generic `ActionDecoder` vocabulary rather than the evolved readout itself.

## Remap strategy

- Strategy: `signed-first-lane`
- `readout[0] > 0` -> `execute`
- `readout[0] < 0` -> `retry`
- `readout[0] == 0` -> `query`
- Expected actions remain `execute`, `query`, `retry`.

## Source summary

| Source | Fitness variant | Cases | Original overall | Remapped overall | Original unexpected | Remapped unexpected |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| benchmarks/results/cunxon_aigarth_action_seed_sweep.json | unknown | 30 | 0.600000 | 0.766667 | 4 | 0 |
| benchmarks/results/cunxon_aigarth_action_hard_holdout_probe.json | unknown | 75 | 0.520000 | 0.506667 | 1 | 0 |
| benchmarks/results/cunxon_aigarth_action_strict_label_probe.json | strict_label_margin | 75 | 0.626667 | 0.586667 | 5 | 0 |
| benchmarks/results/cunxon_aigarth_action_contract_penalty_probe.json | strict_label_heavy_penalty | 75 | 0.480000 | 0.426667 | 3 | 0 |

## Aggregate action distribution

- Original: assertive=13, execute=60, query=118, retry=64
- Remapped: execute=71, query=110, retry=74
- Original unexpected action count: 13
- Remapped unexpected action count: 0

## Accuracy replay

| Split | Original accuracy | Remapped accuracy | Delta |
| --- | ---: | ---: | ---: |
| hard_holdout | 0.555556 | 0.544444 | -0.011111 |
| holdout | 0.583333 | 0.600000 | +0.016667 |
| overall | 0.549020 | 0.537255 | -0.011765 |
| permuted_control | 0.000000 | 0.066667 | +0.066667 |
| train | 0.916667 | 0.816667 | -0.100000 |

## Interpretation boundary

Eliminating out-of-contract labels by remapping is not automatically an adapter improvement. If accuracy falls on train/holdout/hard-holdout splits, the generic decoder was not the only bottleneck. This remains decoder-contract diagnostics, not intelligence, broad generalization, or useful-learning evidence.

## Notes

- post-hoc diagnostic over existing live cuNxon Aigarth artifacts
- signed-first-lane remap follows the project target_readout contract: + first lane=execute, - first lane=retry, neutral=query
- does not create new cuNxon learning evidence; it only isolates decoder vocabulary effects
