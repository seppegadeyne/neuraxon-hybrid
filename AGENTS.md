# AGENTS.md

Project-level instructions for AI agents working in this repository.

## Project overview

`neuraxon-agent` is an experimental integration layer around Neuraxon v2.0. It explores whether Neuraxon-style bio-inspired, trinary, continuous-time neural dynamics can become useful "intelligence tissue" for CLI AI agents.

This is an experimental research/build project. Agents have broad freedom to discover, prototype, instrument, benchmark, and build on top of Neuraxon, as long as changes remain verifiable, reversible through Git, and honest about what the evidence does or does not prove.

Do not overclaim intelligence, learning, generalization, or production readiness. Treat strong claims as benchmark hypotheses that must be backed by artifacts, baselines, holdout/generalization tests, and clear reports.

## Language policy

All repository documentation and Markdown must be written in English. Treat every `*.md` file in this repository as English-only unless it quotes an upstream source or preserves an exact external/API value.

This includes:

- `README.md`, `AGENTS.md`, and all other root docs.
- All Markdown files under `benchmarks/`, `docs/`, `upstream/`, or future documentation folders.
- Generated benchmark reports and diagnostic Markdown artifacts.
- Issue/PR text that is committed into the repo or used as durable project documentation.

Keep code identifiers, API fields, filenames, upstream names, CLI flags, and existing public contracts unchanged unless the task explicitly requires a code/API migration. For example, do not rename a field such as `actie_type` only because the surrounding documentation is English.

## Agent autonomy and research boundaries

Agents may proactively:

- Inspect the codebase and benchmark artifacts.
- Add or refine tests before changing behavior.
- Build adapters, probes, instrumentation, and benchmark reports.
- Run focused and full test suites.
- Create conservative evidence reports that separate runtime viability, adapter behavior, and raw Neuraxon dynamics.
- Explore cuNxon/GPU/native-backend integration when the environment supports it.

Agents must:

- Preserve the distinction between explicit adapter logic and raw Neuraxon network behavior.
- Compare against trivial baselines before calling a result useful.
- Use holdout/generalization tests before making broader claims.
- Keep memory persistence and visual perception deferred unless the decision layer has demonstrated useful baseline-beating behavior that is worth preserving or expanding.
- Avoid editing vendored/upstream reference code unless the task explicitly requires a patch; prefer local adapters, wrappers, or patch notes.

## Working conventions

- Read this file before making changes.
- Check `git status --short --branch` before editing.
- Prefer test-driven changes for behavior and content-contract tests for durable reports/docs.
- Keep Markdown and docs in English.
- Keep benchmark/report language conservative and evidence-based.
- Do not store secrets or local credential values in the repo.
- Do not commit generated caches such as `__pycache__`, `.pytest_cache`, `.mypy_cache`, or `.ruff_cache`.

## Useful commands

Install for development:

```bash
pip install -e ".[dev]"
```

Run the full test suite:

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

Run lint/type checks when relevant:

```bash
ruff check .
mypy src
```

Check Markdown/doc diffs before committing:

```bash
git diff --check
```

## Important paths

- `src/neuraxon_agent/` — project package and agent integration layer.
- `src/neuraxon_agent/vendor/` — vendored/reference Neuraxon code.
- `tests/` — pytest suite.
- `benchmarks/` — benchmark scenarios, results, reports, diagnostics, and analysis artifacts.
- `upstream/` — place for external upstream source snapshots or clones; upstream code remains reference material.

## Commit and verification expectations

Before committing:

1. Run `git diff --check`.
2. Run the focused tests for touched behavior or report contracts.
3. Run the full test suite when feasible: `PYTHONPATH=src python -m pytest tests/ -q`.
4. Review the diff for accidental language drift, stale Dutch docs, overclaims, or unrelated cache files.

Commit messages should be concise and project-scoped, for example:

```text
docs: clarify agent handbook
benchmarks: add temporal holdout report
```
