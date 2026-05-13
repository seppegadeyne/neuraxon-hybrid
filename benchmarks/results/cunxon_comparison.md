# cuNxon comparison against existing Neuraxon-Hybrid evidence

Verdict: `long-horizon runtime viable, not benchmark-integrated`

## What changed

The guarded Hybrid-side `ctypes` smoke path loads the upstream `libcunxon.so` on Aorus and emits a non-empty trinary readout on the RTX 5090. This promotes cuNxon from `build-only` to `smoke-test viable` for a minimal one-sphere runtime smoke.

A second `neuraxon-agent cunxon-long-horizon` probe now keeps one cuNxon network instance active for 65536 steps and samples it every 8192 steps. This explicitly reflects the working assumption that Neuraxon is brain-like and time-dependent: short smoke tests are insufficient for judging whether it learns. In this first long-horizon probe the readout remained [0, 0, 0] across 8 samples (`unique_readouts=1`, `readout_change_count=0`) while energy increased by about 1119554.053.

This is still not a benchmark-integrated backend. The cuNxon lane has no decision-quality score measured yet, and GPU execution does not prove intelligence, generalization, or usefulness by itself.

## Comparison lanes

| Lane | Metric | Value | Evidence |
| --- | --- | ---: | --- |
| cuNxon raw CUDA smoke | decision-quality score | no decision-quality score measured | `benchmarks/results/cunxon_smoke.json` |
| cuNxon long-horizon raw dynamics | continuous runtime samples | 65536 steps; 1 unique readout(s); 0 readout changes | `benchmarks/results/cunxon_long_horizon.json` |
| raw_network | success_rate | 0.145833 | `holdout_noisy_generalization.json` anti-oracle temporal mode |
| random | success_rate | 0.104167 | `holdout_noisy_generalization.json` anti-oracle temporal baseline |
| always_execute | success_rate | 0.166667 | `holdout_noisy_generalization.json` anti-oracle temporal baseline |
| semantic_bridge | success_rate | 1.000000 | explicit semantic bridge mode |
| temporal_context_adapter | success_rate | 1.000000 | explicit temporal context adapter mode |

## Interpretation

cuNxon is viable enough to investigate as a raw CUDA runtime backend, including longer continuous runs. The first long-horizon probe is important because it avoids judging a learning system from only a momentary smoke test. But the observed constant readout means this specific one-sphere probe should be treated as runtime/dynamics evidence, not useful learning evidence.

The existing high scores remain adapter/bridge evidence, while raw continuous-time dynamics stay separated through the `raw_network` ablation line. The next research step is to run longer, richer multi-sphere/task-coupled probes and then map cuNxon readout to the same Neuraxon-Hybrid action contract.

## Evidence boundary

This comparison deliberately separates runtime viability from decision quality. The cuNxon raw CUDA smoke and long-horizon probe prove library loading, GPU stepping, valid trinary readouts, and observable energy dynamics over time. They do not prove intelligence, generalization, or usefulness by themselves.
