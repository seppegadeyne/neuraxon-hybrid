# cuNxon comparison against existing Neuraxon-Hybrid evidence

Verdict: `task-coupled but not above baselines`

## What changed

The guarded Hybrid-side `ctypes` smoke path loads the upstream `libcunxon.so` on Aorus and emits a non-empty trinary readout on the RTX 5090. This promotes cuNxon from `build-only` to `smoke-test viable` for a minimal one-sphere runtime smoke.

A second `neuraxon-agent cunxon-long-horizon` probe keeps one cuNxon network instance active for 65536 steps and samples it every 8192 steps. This explicitly reflects the working assumption that Neuraxon is brain-like and time-dependent: short smoke tests are insufficient for judging whether it learns. In this first long-horizon probe the readout remained [0, 0, 0] across 8 samples (`unique_readouts=1`, `readout_change_count=0`) while energy increased by about 1119554.053.

A third `neuraxon-agent cunxon-action-probe` now runs a tiny task-coupled suite. Each input drive has an expected benchmark action, and the cuNxon trinary readout is decoded through the existing ActionDecoder/action-contract mapping. This is the first cuNxon lane with decision-quality measured, but not useful: it scored 1/3 = 0.333333, with failures on execute/retry drives and success only on the neutral/query drive.

## Comparison lanes

| Lane | Metric | Value | Evidence |
| --- | --- | ---: | --- |
| cuNxon raw CUDA smoke | decision-quality score | no decision-quality score measured | `benchmarks/results/cunxon_smoke.json` |
| cuNxon long-horizon raw dynamics | continuous runtime samples | 65536 steps; 1 unique readout(s); 0 readout changes | `benchmarks/results/cunxon_long_horizon.json` |
| cuNxon task-coupled action probe | success_rate | 0.333333 | `benchmarks/results/cunxon_action_probe.json` |
| raw_network | success_rate | 0.145833 | `holdout_noisy_generalization.json` anti-oracle temporal mode |
| random | success_rate | 0.104167 | `holdout_noisy_generalization.json` anti-oracle temporal baseline |
| always_execute | success_rate | 0.166667 | `holdout_noisy_generalization.json` anti-oracle temporal baseline |
| semantic_bridge | success_rate | 1.000000 | explicit semantic bridge mode |
| temporal_context_adapter | success_rate | 1.000000 | explicit temporal context adapter mode |

## Interpretation

cuNxon is viable enough to investigate as a raw CUDA runtime backend, including longer continuous runs and task-coupled probes. The new task-coupled action probe is more meaningful than smoke/runtime evidence because it maps GPU readouts to the same action contract used by Neuraxon-Hybrid benchmarks.

The result is still negative/diagnostic rather than promising: decision-quality measured but not useful. Accuracy is baseline-level for a three-action toy suite, and the readouts remain too flat/weak to support a learning or generalization claim. The next research step should be richer multi-sphere probes, stronger input encoding, and direct comparison against trivial policies on holdout temporal scenarios.

The existing high scores remain adapter/bridge evidence, while raw continuous-time dynamics stay separated through the `raw_network` ablation line.

## Evidence boundary

This comparison deliberately separates runtime viability from decision quality. The cuNxon raw CUDA smoke and long-horizon probe prove library loading, GPU stepping, valid trinary readouts, and observable energy dynamics over time. The task-coupled action probe measures a small decision-quality surface, but baseline-level performance does not prove intelligence, generalization, or usefulness by itself.
