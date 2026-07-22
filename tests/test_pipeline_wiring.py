import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

UV_STUB = """#!/usr/bin/env bash
echo "[uv] $*"
for a in "$@"; do
  case "$a" in
    *05_evaluate.py) mkdir -p eval/results; echo table > eval/results/results.md; exit {gate_rc} ;;
    *06_push_to_hub.py) echo "PUBLISH_CALLED" ;;
  esac
done
exit 0
"""


def _sandbox(tmp_path, gate_rc):
    """A fake repo where every heavy step is a stub, so the control flow can be run."""
    (tmp_path / "scripts").mkdir()
    shutil.copy("scripts/run_pipeline.sh", tmp_path / "scripts" / "run_pipeline.sh")
    (tmp_path / "scripts" / "04_quantize.sh").write_text('echo "[stub] quantize"\n')
    binx = tmp_path / "bin"
    binx.mkdir()
    uv = binx / "uv"
    uv.write_text(UV_STUB.format(gate_rc=gate_rc))
    uv.chmod(0o755)
    pip = binx / "pip"
    pip.write_text("#!/usr/bin/env bash\nexit 0\n")
    pip.chmod(0o755)
    return binx


def _run_pipeline(tmp_path, gate_rc, max_steps):
    binx = _sandbox(tmp_path, gate_rc)
    import os

    env = {**os.environ, "PATH": f"{binx}:{os.environ['PATH']}", "HF_ORG": "x", "TAG": "v0"}
    if max_steps is not None:
        env["MAX_STEPS"] = max_steps
    return subprocess.run(
        ["bash", "scripts/run_pipeline.sh"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )


def test_full_run_publishes_when_the_gate_passes(tmp_path):
    # Guards the production path: it must survive `set -u` even though the smoke-only
    # eval arguments are empty here.
    res = _run_pipeline(tmp_path, gate_rc=0, max_steps=None)
    assert "unbound variable" not in res.stderr
    assert res.returncode == 0, res.stdout + res.stderr
    assert "PUBLISH_CALLED" in res.stdout


def test_full_run_aborts_without_publishing_when_the_gate_fails(tmp_path):
    res = _run_pipeline(tmp_path, gate_rc=1, max_steps=None)
    assert res.returncode == 1
    assert "PUBLISH_CALLED" not in res.stdout


@pytest.mark.parametrize("gate_rc", [0, 1])
def test_smoke_run_never_publishes_regardless_of_the_gate(tmp_path, gate_rc):
    res = _run_pipeline(tmp_path, gate_rc=gate_rc, max_steps="10")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "PUBLISH_CALLED" not in res.stdout
    assert "--max-steps 10" in res.stdout
    assert "--limit" in res.stdout


def test_non_numeric_max_steps_is_rejected(tmp_path):
    res = _run_pipeline(tmp_path, gate_rc=0, max_steps="abc")
    assert res.returncode == 2
    assert "PUBLISH_CALLED" not in res.stdout


def test_release_workflow_is_manual_dispatch_with_inputs():
    wf = yaml.safe_load(Path(".github/workflows/release.yml").read_text(encoding="utf-8"))
    # PyYAML parses the bare `on:` key as boolean True
    triggers = wf.get("on", wf.get(True))
    assert "workflow_dispatch" in triggers
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert {"flavor", "tag", "ref"} <= set(inputs)


def test_release_workflow_exposes_max_steps_input():
    wf = yaml.safe_load(Path(".github/workflows/release.yml").read_text(encoding="utf-8"))
    triggers = wf.get("on", wf.get(True))
    assert "max_steps" in triggers["workflow_dispatch"]["inputs"]


def test_release_workflow_defaults_to_the_cheapest_bf16_capable_flavor():
    # T4 is cheaper but is Turing and has no bf16, which the merge step requires.
    # L4 is the cheapest flavor that runs the pipeline unmodified.
    wf = yaml.safe_load(Path(".github/workflows/release.yml").read_text(encoding="utf-8"))
    triggers = wf.get("on", wf.get(True))
    assert triggers["workflow_dispatch"]["inputs"]["flavor"]["default"] == "l4x1"


def test_release_workflow_forwards_max_steps_to_the_job():
    text = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "MAX_STEPS" in text


def test_run_pipeline_forwards_max_steps_to_training():
    src = Path("scripts/run_pipeline.sh").read_text(encoding="utf-8")
    assert "MAX_STEPS" in src
    assert "--max-steps" in src


def test_run_pipeline_smoke_mode_limits_eval_and_never_publishes():
    # A smoke run trains a deliberately useless model, so the eval gate is expected to
    # fail. That must not be reported as a broken pipeline, and must never publish.
    src = Path("scripts/run_pipeline.sh").read_text(encoding="utf-8")
    assert "--limit" in src
    smoke = src[src.index("SMOKE") :]
    assert "06_push_to_hub.py" not in smoke.split("exit 0")[0]


def test_release_workflow_runs_jobs_in_the_org_namespace():
    # Without --namespace, `hf jobs run` uses the token owner's personal namespace,
    # which both fails on an org-scoped token and bills the wrong account.
    text = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "--namespace" in text


def test_preflight_launches_a_real_job_to_prove_write_permission():
    # `hf jobs ls` is a READ and succeeds without job.write, so listing alone cannot
    # tell us whether the pipeline may launch. Only an actual launch proves it.
    text = Path(".github/workflows/preflight.yml").read_text(encoding="utf-8")
    assert "hf jobs run" in text
    assert "cpu-basic" in text
    assert "--namespace" in text


def test_eval_gate_runs_tests_and_lint():
    text = Path(".github/workflows/eval-gate.yml").read_text(encoding="utf-8")
    assert "pytest" in text and "ruff" in text


def test_run_pipeline_script_has_full_chain_in_order():
    src = Path("scripts/run_pipeline.sh").read_text(encoding="utf-8")
    order = [
        "01_build_dataset.py",
        "02_train.py",
        "03_merge.py",
        "04_quantize.sh",
        "05_evaluate.py",
        "06_push_to_hub.py",
    ]
    positions = [src.index(step) for step in order]
    assert positions == sorted(positions), "pipeline steps must appear in order"
    assert "Q4_K_M" in src
    assert "set -euo pipefail" in src
