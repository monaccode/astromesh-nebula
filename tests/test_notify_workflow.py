from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]


def test_notify_workflow_parses_and_targets_lock():
    wf = yaml.safe_load((_ROOT / ".github/workflows/notify-catalog.yml").read_text())
    # PyYAML parses the bare key `on:` as the boolean True (YAML 1.1), not "on".
    push = wf.get("on", wf.get(True))["push"]
    assert "nebula/catalog.lock.json" in push["paths"]
    step_runs = " ".join(str(s.get("run", "")) for s in wf["jobs"]["notify"]["steps"])
    assert "catalog-lock-updated" in step_runs
    assert "dispatches" in step_runs
