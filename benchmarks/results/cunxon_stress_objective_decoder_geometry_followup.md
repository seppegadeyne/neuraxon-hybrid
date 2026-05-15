# cuNxon stress objective decoder/readout geometry follow-up

Issue: https://github.com/sisutuulenisa/neuraxon-hybrid/issues/89

## Hypothesis

The latest target-aligned stress objective preserved the high-amplitude `3.0x` stress signal, but the original low-margin `stress_holdout` split still collapsed to `query`. This artifact asks whether the boundary is best explained as signed-first-lane decoder/readout geometry rather than by missing stress labels, criticality estimator movement, or insufficient runtime.

## Source artifacts

- `benchmarks/results/cunxon_aigarth_action_target_contract_stress_objective_probe.json`
- `benchmarks/results/cunxon_aigarth_action_target_contract_stress_amplitude_ladder_probe.json`
- `benchmarks/results/cunxon_stress_geometry_audit.json`

## Result

- Original stress_holdout accuracy: `0.333333`
- Original stress_holdout query-collapse: `1.000000`
- Original stress_holdout execute/retry accuracy: `0.000000`
- Scaled stress_holdout accuracy: `0.888889`
- Scaled stress_holdout query-collapse: `0.222222`
- Scaled stress_holdout execute/retry accuracy: `1.000000`
- Best constant baseline: `0.333333`
- Original delta vs best constant: `0.000000`
- Scaled delta vs best constant: `0.555556`

## Geometry summary

- Original execute/retry first-lane abs mean: `0.150000`
- Scaled execute/retry first-lane abs mean: `0.450000`
- first-lane gain: `3.000000`
- Original execute/retry L2 mean: `0.209134`
- Scaled execute/retry L2 mean: `0.627402`

| Split | Accuracy | Query-collapse | Execute/retry accuracy | Action distribution |
| --- | ---: | ---: | ---: | --- |
| original stress_holdout | `0.333333` | `1.000000` | `0.000000` | `query=18` |
| scaled stress_holdout 3.0x | `0.888889` | `0.222222` | `1.000000` | `execute=6, query=4, retry=8` |

## Interpretation

The post-hoc geometry supports a narrow conclusion: the signed-first-lane route can separate the same stress shapes after a 3.0x amplitude gain, but the original low-margin split remains exactly at the best constant baseline and fully collapsed to `query`. This is decoder/readout geometry evidence, not intelligence evidence and not generalization evidence.

## Recommended next probe

`low_margin_readout_geometry_probe`: only if the next run changes the lower-level readout hypothesis, inspect a normalized or contrast-coded low-margin route while keeping original `stress_holdout` and controls outside any optimized labels. Do not repeat the same margin-weighted 3.0x objective longer.
