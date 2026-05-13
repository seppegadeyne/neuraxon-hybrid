# cuNxon comparison against existing Neuraxon-Hybrid evidence

Verdict: `sensitivity diagnostic train-mode flat`

## What changed

The guarded Hybrid-side `ctypes` smoke path loads the upstream `libcunxon.so` on Aorus and emits a non-empty trinary readout on the RTX 5090. This promotes cuNxon from `build-only` to `smoke-test viable` for a minimal one-sphere runtime smoke.

A second `neuraxon-agent cunxon-long-horizon` probe keeps one cuNxon network instance active for 65536 steps and samples it every 8192 steps. This explicitly reflects the working assumption that Neuraxon is brain-like and time-dependent: short smoke tests are insufficient for judging whether it learns. In this first long-horizon probe the readout remained [0, 0, 0] across 8 samples (`unique_readouts=1`, `readout_change_count=0`) while energy increased by about 1119554.053.

A third `neuraxon-agent cunxon-action-probe` runs a tiny task-coupled suite. Each input drive has an expected benchmark action, and the cuNxon trinary readout is decoded through the existing ActionDecoder/action-contract mapping. This is the first cuNxon lane with decision-quality measured, but not useful: it scored 1/3 = 0.333333, with failures on execute/retry drives and success only on the neutral/query drive.

A fourth `neuraxon-agent cunxon-sensitivity-probe` compares frozen `cunxonNetworkStepInfer` with paper-canonical plastic `cunxonNetworkStepTrain` using fresh one-sphere networks for 3 stimuli x 3 seed offsets x 2 modes. The result is diagnostic and negative: infer mode has 3 unique readouts but still decodes mostly to `query`; train mode is completely flat (`unique_readouts=1`, `query=9/9`). This suggests the current one-sphere train-mode setup suppresses rather than improves task-coupled action signal.

## Comparison lanes

| Lane | Metric | Value | Evidence |
| --- | --- | ---: | --- |
| cuNxon raw CUDA smoke | decision-quality score | no decision-quality score measured | `benchmarks/results/cunxon_smoke.json` |
| cuNxon long-horizon raw dynamics | continuous runtime samples | 65536 steps; 1 unique readout(s); 0 readout changes | `benchmarks/results/cunxon_long_horizon.json` |
| cuNxon task-coupled action probe | success_rate | 0.333333 | `benchmarks/results/cunxon_action_probe.json` |
| cuNxon infer-vs-train sensitivity probe | readout/action diversity | infer unique=3; train unique=1; train-mode flat/query=9 | `benchmarks/results/cunxon_sensitivity_probe.json` |
| raw_network | success_rate | 0.145833 | `holdout_noisy_generalization anti-oracle temporal mode` |
| random | success_rate | 0.104167 | `holdout_noisy_generalization anti-oracle temporal baseline` |
| always_execute | success_rate | 0.166667 | `holdout_noisy_generalization anti-oracle temporal baseline` |
| semantic_bridge | success_rate | 1.000000 | `holdout_noisy_generalization semantic bridge mode` |
| temporal_context_adapter | success_rate | 1.000000 | `holdout_noisy_generalization temporal adapter mode` |

## Interpretation

cuNxon is viable enough to investigate as a raw CUDA runtime backend, including longer continuous runs, task-coupled probes and infer-vs-train diagnostics. The new sensitivity probe answers the immediate concern that the previous action probe used frozen inference rather than the paper-canonical continuous-learning step.

The result remains negative/diagnostic: decision-quality measured but not useful, and train-mode flat for this minimal one-sphere setup. It does not support a useful raw GPU-backed policy or learning claim. The next research step should not be another smoke test; it should either inspect richer cuNxon surfaces such as snapshots/pattern APIs or build a deliberately richer multi-sphere/action adapter with direct holdout temporal comparison against trivial policies.

The existing high scores remain adapter/bridge evidence, while raw continuous-time dynamics stay separated through the `raw_network` ablation line.

## Evidence boundary

This comparison deliberately separates runtime viability from decision quality. The cuNxon raw CUDA smoke and long-horizon probe prove library loading, GPU stepping, valid trinary readouts and observable energy dynamics over time. The task-coupled action probe measures a small decision-quality surface, and the sensitivity probe compares infer/train dynamics, but flat or baseline-level performance does not prove intelligence, generalization, or usefulness by itself.
