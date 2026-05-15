# cuNxon source semantics audit

Status: `source semantics audited`

Active issue: <https://github.com/sisutuulenisa/neuraxon-hybrid/issues/80>

This audit inspected the upstream cuNxon C/CUDA source and re-ran the upstream 4-sphere CUDA example on Aorus RTX 5090. It is interface/runtime evidence, not a decision-quality improvement, and it does not prove intelligence.

## Hypothesis

The supervised motor-target probe stayed flat because absolute output-neuron target ports are valid readout ports, but they are not a supported desired-output or teacher-forcing channel in the cuNxon `StepTrain` API.

## Source findings

| Finding | Source | Implication |
| --- | --- | --- |
| Public execution API exposes `cunxonNetworkStepInfer`, `cunxonNetworkStepTrain`, `cunxonNetworkRun`, neuromodulator injection and readout copy, but no desired-output API surface. | `include/cuNxon.h:173-220` | Reward/neuromodulator shaping is the exposed route; there is no label/loss/error-vector argument for supervised motor targets. |
| External inputs are scattered only onto `sensory_input_ids`. | `src/cuNxon.cu:880-903` | A step input buffer is mapped by the interface's sensory port list, not by arbitrary output readout semantics. |
| External input only directly drives input-class neurons. | `src/cuNxon_kernels.cu:422-433,503-508` | If an output-neuron absolute id is put in `sensory_input_ids`, the value is copied to `ext_in[id]`, but output neurons do not consume that as direct teacher-forced state. |
| CTSN online learning targets firing-rate homeostasis. | `src/cuNxon_kernels.cu:591-608` | `target_firing_rate` is not a supervised desired-output label; it tunes `phi_gain`/`phi_bias` toward activity rate. |
| `cunxonSphereGetReadout` gathers `N.s[port_out_readout[i]]`. | `src/cuNxon.cu:1090-1110` | Readout is a copied trinary state slice, not a trainable supervised readout/error head. |
| C++ example uses absolute output-neuron port ids. | `examples/example_4sphere.cu:192-256` | The prior interface-semantics probe matches the intended C++ index convention: output block starts at `n_input + n_hidden`. |
| Python example still uses relative output-port ids. | `examples/python_binding.py:533-556` | The Python demo should not be used as adapter guidance until fixed; those relative ids select input neurons under the C++ API convention. |

## Live upstream runtime check

Command:

```bash
./example_4sphere 3000 400 0
```

Working directory:

```text
/home/seppe/Projects/research/DavidVivancos-Neuraxon/cuNxon/build-hermes-issue79-20260513140153
```

Observed result on Aorus RTX 5090:

```text
overall accuracy              : 48.2%  (chance = 50.0%)
```

Details:

- Warmup: 3,000 `StepTrain` steps.
- Online test: 400 additional `StepTrain` steps with plasticity still active, matching the upstream example's continuous-learning framing.
- Salient detection: 98 / 216 (45.4%).
- Non-salient rejection: 95 / 184 (51.6%).
- Overall accuracy: 193 / 400 = 0.4825.
- Energy and activity changed throughout the run, but this did not produce above-chance decision quality in this run.

## Interpretation

The negative supervised-target result is now less mysterious: the route used by the Hybrid probe copied teacher-forcing values into `ext_in` slots named by `sensory_input_ids`, but the cuNxon membrane kernel only treats input-class neurons as direct external pass-through. Output neurons are readout-capable and can be used as inter-sphere relay/readout ports, but they are not directly target-forced by the public step input API.

This means the current evidence should not be read as "cuNxon failed supervised learning in general". It is narrower: the tested output-port teacher-forcing adapter is unsupported by source-level API semantics and remained flat/baseline-level in live probes.

## Evidence boundary

This audit strengthens the conservative boundary for issue #79/#80:

- Runtime viability: still supported.
- Interface semantics: absolute output-neuron readout ids are supported in the C++ path.
- Output-target route: no desired-output API surface found; output-port teacher forcing through `sensory_input_ids` is not supported by the kernel path.
- Decision quality: upstream reward-only 4-sphere example ran but scored 48.2% in this live run, below its 50.0% chance reference; Hybrid action probes remain flat or baseline-level.
- Claim boundary: this does not prove intelligence, useful learning or generalization.

## Next useful hypothesis

Do not rerun the same supervised output-port target adapter. If continuing this lane, test one of these instead:

1. A lower-level input-port supervised proxy where target information drives actual input-class neurons and readout is evaluated separately.
2. An upstream patch/probe that adds or exposes a real desired-output/error-vector route, if that matches cuNxon's intended learning semantics.
3. A VRAM-resident dynamics run only after defining dynamics observables that are not already answered by the flat output-target route.
