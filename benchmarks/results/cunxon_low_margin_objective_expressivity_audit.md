# cuNxon low-margin objective expressivity audit

Issue: https://github.com/sisutuulenisa/neuraxon-hybrid/issues/80

## Hypothesis

After the supervised low-margin objective, the useful question is no longer whether normalized low-margin examples are separable. They are. The question is whether the current cuNxon public surfaces can express the original low-margin stress/control boundary without leaking those labels into optimization.

## Source and artifact evidence

- `supervised_low_margin_train` accuracy is `1.000000` and execute/retry accuracy is `1.000000`.
- Original stress_holdout remains `0.333333` with query-collapse `1.000000` and execute/retry accuracy `0.000000`.
- Counterfactual and permuted controls remain `0.000000` and `0.000000`.
- Source inspection shows Aigarth is scalar black-box selection: `cunxonFitnessFn_t` returns one float, `cunxonNetworkAigarthStep` mutates weights, scores candidates, installs the best weights, and resets dynamic state.
- The desired-output API is absent in the public `StepTrain` route; previous source/runtime artifacts show external drive is input-class stimulation only, not hidden/output teacher forcing.

## Diagnosis

`normalized_surrogate_learned_original_low_margin_boundary_unchanged`.

The normalized low-margin target route is learnable inside the scalar Aigarth fitness callback, but that does not translate to the original low-margin stress split. The likely boundary is objective/readout expressivity: the API supports black-box scoring of whole candidate networks and input-class external drive, but not a direct per-output desired-label/error channel.

## Evidence boundary

This is not intelligence evidence and not generalization evidence. It is a post-hoc source-and-artifact audit that explains why another longer copy of the same supervised objective is unlikely to be the next useful slice.

## Recommended next probe

`readout_margin_objective_or_ctsn_semantics_audit`: inspect readout threshold/CTSN target-firing-rate semantics or an explicitly engineered readout-margin objective while keeping original stress/control labels outside optimization. Acceptance still requires original stress/control quality above constant baselines before any useful-computation claim.
