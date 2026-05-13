"""Tests for neuraxon-agent CLI."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from neuraxon_agent.cli import main
from neuraxon_agent.cunxon_smoke import CunxonSmokeResult


def test_cli_help() -> None:
    assert main(["--help"]) == 0


def test_cli_think() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = Path(tmpdir) / "obs.json"
        out = Path(tmpdir) / "act.json"
        inp.write_text(json.dumps({"observation": {"type": "prompt", "content": "hi"}}))
        rc = main(["think", "-i", str(inp), "-o", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert "action" in data
        assert "confidence" in data


def test_cli_modulate() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = Path(tmpdir) / "out.json"
        out = Path(tmpdir) / "res.json"
        inp.write_text(json.dumps({"outcome": "success"}))
        rc = main(["modulate", "-i", str(inp), "-o", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert "state" in data


def test_cli_evolve() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "evo.json"
        rc = main(["evolve", "-g", "1", "-e", "1", "--seed", "42", "-o", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert "summary" in data


def test_cli_cunxon_smoke_writes_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_smoke(**_: object) -> CunxonSmokeResult:
        return CunxonSmokeResult(
            status="smoke-test viable",
            upstream_commit="upstream",
            cunxon_commit="cunxon",
            library_path="/tmp/libcunxon.so",
            device_name="NVIDIA GeForce RTX 5090",
            compute_capability="12.0",
            steps=3,
            readout=[-1, 0, 1],
            energy=0.5,
            elapsed_ms=1.25,
            notes=["fake smoke"],
        )

    monkeypatch.setattr("neuraxon_agent.cli.run_ctypes_smoke", fake_smoke)
    json_path = tmp_path / "cunxon.json"
    markdown_path = tmp_path / "cunxon.md"

    rc = main(
        [
            "cunxon-smoke",
            "--library",
            "/tmp/libcunxon.so",
            "--upstream-commit",
            "upstream",
            "--cunxon-commit",
            "cunxon",
            "--steps",
            "3",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert rc == 0
    assert '"status": "smoke-test viable"' in json_path.read_text(encoding="utf-8")
    assert "No broad Neuraxon intelligence claim" in markdown_path.read_text(encoding="utf-8")


def test_cli_no_command() -> None:
    assert main([]) == 2
