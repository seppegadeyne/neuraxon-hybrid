# cuNxon comparison against existing Neuraxon-Hybrid evidence

Verdict: `smoke-test viable, not benchmark-integrated`

## What changed

The guarded Hybrid-side `ctypes` smoke path now loads the upstream `libcunxon.so` on Aorus and emits a non-empty trinary readout on the RTX 5090. This promotes cuNxon from `build-only` to `smoke-test viable` for a minimal one-sphere runtime smoke.

This is still not a benchmark-integrated backend. The cuNxon lane has no decision-quality score measured yet, and GPU execution does not prove intelligence, generalization, or usefulness by itself.

## Comparison lanes

| Lane | Metric | Value | Evidence |
| --- | --- | ---: | --- |
| cuNxon raw CUDA smoke | decision-quality score | no decision-quality score measured | `benchmarks/results/cunxon_smoke.json` |
| raw_network | success_rate | 0.145833 | `holdout_noisy_generalization.json` anti-oracle temporal mode |
| random | success_rate | 0.104167 | `holdout_noisy_generalization.json` anti-oracle temporal baseline |
| always_execute | success_rate | 0.166667 | `holdout_noisy_generalization.json` anti-oracle temporal baseline |
| semantic_bridge | success_rate | 1.000000 | explicit semantic bridge mode |
| temporal_context_adapter | success_rate | 1.000000 | explicit temporal context adapter mode |

## Interpretation

cuNxon is now viable enough to investigate as a raw CUDA runtime backend, but it has not beaten simple baselines on the existing decision tasks because it is not wired into those tasks yet. The existing high scores remain adapter/bridge evidence, while raw continuous-time dynamics stay separated through the `raw_network` ablation line.

Next research step: define a small cuNxon-backed decision adapter or raw-dynamics probe that maps the GPU readout to the same benchmark action contract, then compare it directly against `raw_network`, `random`, and `always_execute` without treating speed as intelligence evidence.

## Evidence boundary

This comparison deliberately separates runtime viability from decision quality. The cuNxon raw CUDA smoke only proves library loading, GPU stepping, and valid trinary readout. It does not prove intelligence, generalization, or usefulness by itself.
