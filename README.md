# neuraxon-agent

Intelligence Tissue for CLI AI Agents, powered by [Neuraxon](https://github.com/DavidVivancos/Neuraxon) v2.0.

## What is this?

`neuraxon-agent` is an agent integration layer around Neuraxon v2.0 — a bio-inspired neural network with trinary states, continuous time, and 4 neuromodulators. This project makes it possible to use Neuraxon as "intelligence tissue" for CLI AI agents such as Hermes.

## Architecture

```
AgentTissue (wrapper)
  ├── PerceptionEncoder    → converts observations into trinary input
  ├── NeuraxonNetwork      → the bio-neural network
  ├── ActionDecoder        → converts output into agent actions
  ├── ModulationFeedback   → dopamine/serotonin feedback loop
  ├── TissueMemory         → pattern storage and recall
  ├── AgentEvolution       → Aigarth evolutionary training
  └── StreamingLoop        → real-time simulation loop
```

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Use the CLI
neuraxon-agent think -i observation.json -o action.json
neuraxon-agent modulate -i outcome.json -o result.json
neuraxon-agent evolve -g 5 -o summary.json

# Python API
from neuraxon_agent import AgentTissue

tissue = AgentTissue()
tissue.observe({"type": "prompt", "content": "hello"})
action = tissue.think(steps=10)
print(action.actie_type, action.confidence)
tissue.modulate("success")
```

## Project structure

| Module | Purpose |
|--------|---------|
| `perception.py` | Observations → trinary input encoding |
| `action.py` | Trinary output → agent actions |
| `tissue.py` | NeuraxonNetwork wrapper |
| `modulation.py` | Neuromodulator feedback loop |
| `memory.py` | Pattern storage and recall |
| `evolution.py` | Aigarth evolutionary training |
| `streaming.py` | Real-time simulation loop |
| `cli.py` | JSON CLI interface |
| `vendor/` | Neuraxon v2.0 upstream code |

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

## Status

This project is in active development. See the [GitHub issues](https://github.com/sisutuulenisa/neuraxon-hybrid/issues) for the roadmap.

## Roadmap gate

Current phase: benchmark the decision layer before expanding input/state complexity.
The current benchmark report shows useful semantic routing and an explicit
`temporal_context_bridge`, but raw Neuraxon dynamics have not yet demonstrated a
learned policy that generalizes meaningfully above baselines. The near-term
blockers and prerequisite evidence are tracked in:

- #51 — expanded temporal benchmark beyond the original smoke probe.
- #52 — policy-ablation separating semantic-bridge behavior from raw-network behavior.
- #53 — temporal state carry-over into action decisions.
- #54 — criticality and neuromodulator dynamics instrumentation.
- #55 — this roadmap gate and documentation/linking pass.

Minimum evidence before deferred work resumes:

- Memory persistence remains deferred until temporal benchmark performance is
  meaningfully above baselines and the useful behavior survives raw/adapter
  separation, so persisted state would preserve real decision value rather than
  semantic routing artifacts.
- Visual perception remains deferred until the base decision layer generalizes
  beyond hand-authored semantic routing. Additional screenshots, DOM grids, or
  multi-sphere visual inputs should not be added before the core policy can make
  useful non-visual decisions under temporal/stateful evaluation.
