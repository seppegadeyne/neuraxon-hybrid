# cuNxon low-margin readout geometry probe

Issue: https://github.com/sisutuulenisa/neuraxon-hybrid/issues/89

## Hypothesis

The previous stress-objective geometry artifact showed a clean `3.0x` first-lane gain. This probe asks whether the remaining failure is a low-margin readout/decoder boundary: original execute/retry stress vectors may sit on the wrong side of the query boundary, while scaled vectors cross it.

## Source artifacts

- `benchmarks/results/cunxon_stress_objective_decoder_geometry_followup.json`
- `benchmarks/results/cunxon_aigarth_action_target_contract_stress_objective_probe.json`

## Result

- Original stress_holdout accuracy: `0.333333`
- Original stress_holdout query-collapse: `1.000000`
- Original stress_holdout execute/retry accuracy: `0.000000`
- Scaled stress_holdout accuracy: `0.888889`
- Scaled stress_holdout query-collapse: `0.222222`
- Scaled stress_holdout execute/retry accuracy: `1.000000`
- Original first-lane abs mean: `0.150000`
- Scaled first-lane abs mean: `0.450000`
- Diagnostic query-boundary threshold: `0.300000`
- Original first-lane margin to query boundary: `-0.150000`
- Scaled first-lane margin to query boundary: `0.150000`

## Interpretation

The result supports a narrow geometry conclusion: original low-margin execute/retry stress vectors are on the wrong side of the query boundary and still collapse completely to `query`; the same shapes become separable only after the `3.0x` scaling used in the label-injected objective path. This is not intelligence evidence and not generalization evidence.

## Recommended next probe

`supervised_low_margin_target_objective`: test a normalized or supervised low-margin objective that targets original-scale lane margins while keeping original `stress_holdout` and controls outside the optimized labels. Stop if original stress/control remains at constant baselines.
